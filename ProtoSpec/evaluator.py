import inspect
import json
import os
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional

from ProtoSpec import tasks
from ProtoSpec.generation import parallel_generations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_WARNING = """
Code execution is required for this task but is disabled. Set --allow_code_execution to enable.
"""

# Task comparison configuration
TASK_COMPARISON_CONFIG = {
    "ids_auto_complete": {"compare_old": True, "compare_comm": False},
    "ids_startup": {"compare_old": False, "compare_comm": True},
    "ids_edit": {"compare_old": True, "compare_comm": True},
}

class Evaluator:
    """Evaluator for generating and evaluating model outputs on various tasks."""
    
    def __init__(self, accelerator, model, tokenizer, args):
        """Initialize the Evaluator.
        
        Args:
            accelerator: Accelerator instance for distributed computing
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            args: Arguments containing evaluation configuration
        """
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # Setup arguments
        self.metric_output_path = args.metric_output_path

        # Code evaluation permission
        self.allow_code_execution = args.allow_code_execution
        
        logger.info(f"Evaluator initialized with metric output path: {self.metric_output_path}")

    def _calculate_n_tasks(self, dataset_length: int) -> int:
        """Calculate the number of tasks to process based on limit settings.
        
        Args:
            dataset_length: Total length of the dataset
            
        Returns:
            Number of tasks to process
        """
        if self.args.limit:
            n_tasks = min(self.args.limit, dataset_length - self.args.limit_start)
        else:
            n_tasks = dataset_length - self.args.limit_start
        
        logger.info(f"Processing {n_tasks} tasks (dataset length: {dataset_length}, "
                   f"limit_start: {self.args.limit_start}, limit: {self.args.limit})")
        return n_tasks

    def _extract_dataset_fields(
        self, 
        task, 
        dataset, 
        n_tasks: int
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract references, old texts, and comments from dataset.
        
        Args:
            task: Task instance
            dataset: Dataset to extract from
            n_tasks: Number of tasks to process
            
        Returns:
            Tuple of (references, old_texts, comments)
        """
        start_idx = self.args.limit_start
        end_idx = start_idx + n_tasks
        
        references = [task.get_reference(dataset[i]) for i in range(start_idx, end_idx)]
        old_texts = [task.get_old_text(dataset[i]) for i in range(start_idx, end_idx)]
        comments = [task.get_comments(dataset[i]) for i in range(start_idx, end_idx)]
        
        logger.info(f"Extracted {len(references)} references, {len(old_texts)} old texts, "
                   f"and {len(comments)} comments")
        return references, old_texts, comments

    def generate_text(
        self, 
        task_name: str, 
        intermediate_generations: Optional[List] = None
    ) -> Tuple[List[List[str]], List[str], List[str], List[str]]:
        """Generate text for a given task.
        
        Args:
            task_name: Name of the task to generate text for
            intermediate_generations: Optional list of intermediate generations to continue from
            
        Returns:
            Tuple of (generations, references, old_texts, comments)
        """
        logger.info(f"Starting text generation for task: {task_name}")
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        
        n_tasks = self._calculate_n_tasks(len(dataset))
        references, old_texts, comments = self._extract_dataset_fields(task, dataset, n_tasks)

        # Check references mode
        if self.args.check_references:
            logger.info("Check references mode enabled")
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [
                    [task.get_reference(dataset[i], get_solution=True)] 
                    for i in range(self.args.limit_start, self.args.limit_start + n_tasks)
                ]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references, old_texts, comments

        # Handle intermediate generations
        curr_generations = []
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
            logger.info(f"Continuing from {len(curr_generations)} intermediate generations")
        
        intermediate_save_path = (
            f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        )
        curr_sample_idx = len(curr_generations)

        logger.info(f"Generating {n_tasks} new samples (starting from index {curr_sample_idx})")
        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_path,
        )

        # Trim excess generations if needed
        if len(generations[0]) > self.args.n_samples:
            original_count = len(generations[0])
            generations = [gen_list[:self.args.n_samples] for gen_list in generations]
            logger.warning(
                f"Trimmed generations from {original_count} to {self.args.n_samples} samples "
                f"(uneven distribution across devices)"
            )
        
        logger.info(f"Text generation complete for task: {task_name}")
        return generations, references, old_texts, comments

    def _get_comparison_config(self, task_name: str) -> Dict[str, bool]:
        """Get comparison configuration for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with compare_old and compare_comm flags
        """
        return TASK_COMPARISON_CONFIG.get(
            task_name, 
            {"compare_old": False, "compare_comm": False}
        )

    def evaluate(
        self, 
        task_name: str, 
        intermediate_generations: Optional[List] = None, 
        compare_old: bool = False, 
        compare_comm: bool = False
    ) -> Dict[str, Any]:
        """Evaluate generated text on a task.
        
        Args:
            task_name: Name of the task to evaluate
            intermediate_generations: Optional list of intermediate generations
            compare_old: Whether to compare with old text (overridden by task config)
            compare_comm: Whether to compare with comments (overridden by task config)
            
        Returns:
            Dictionary with evaluation results for gold, old, and comments comparisons
        """
        logger.info(f"Starting evaluation for task: {task_name}")
        task = tasks.get_task(task_name, self.args)
        
        # Check code execution permission
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        # Generate text
        generations, references, old_texts, comments = self.generate_text(
            task_name, 
            intermediate_generations=intermediate_generations
        )
        
        if self.accelerator.is_main_process:
            # Save generations if not loading from file
            if not self.args.load_generations_path:
                save_generations_path = (
                    f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                )
                self.save_json_files(
                    generations, 
                    references, 
                    save_generations_path, 
                    f"references_{task_name}.json"
                )

            # Configure environment for evaluation
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            
            # Get task-specific comparison configuration
            config = self._get_comparison_config(task_name)
            compare_old = config["compare_old"]
            compare_comm = config["compare_comm"]
            
            logger.info(f"Evaluation config - compare_old: {compare_old}, compare_comm: {compare_comm}")
            logger.info("Computing evaluation metrics...")

            # Initialize results
            results0, results1, results2 = {}, {}, {}
            
            # Evaluate against gold references
            results0 = task.process_results(generations, references)
            
            # Optionally evaluate against old texts
            if compare_old:
                logger.info("Computing metrics against old text...")
                results1 = task.process_results(generations, old_texts)
            
            # Optionally evaluate against comments
            if compare_comm:
                logger.info("Computing metrics against comments...")
                results2 = task.process_results(generations, comments)
            
            logger.info(f"Evaluation complete for task: {task_name}")
            return {'gold': results0, 'old': results1, 'comments': results2}

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        """Save generations and references to JSON files.
        
        Args:
            generations: List of generated texts to save
            references: List of reference texts to save
            save_generations_path: Path to save generations
            save_references_path: Path to save references
        """
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp, indent=2)
            logger.info(f"Generations saved to: {save_generations_path}")
            
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp, indent=2)
            logger.info(f"References saved to: {save_references_path}")
