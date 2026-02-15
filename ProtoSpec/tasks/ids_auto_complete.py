import json
import re
import os
import logging
import importlib.util
from typing import Dict, List, Any, Tuple
from evaluate import load
from ProtoSpec.base import Task
import csv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ORDER = 4
DEFAULT_SMOOTH = True


def _is_package_available(package_name: str) -> bool:
    """Check if a package is available in the environment.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


def requires_package(
    metric_name: str, package_name: str, install_instruction: str = None
) -> None:
    """Check if a package is available and raise an error with installation instructions if not.
    
    Args:
        metric_name: The name of the metric that requires the package
        package_name: The name of the package to check
        install_instruction: Custom install command. If None, defaults to "pip install {package_name}"
    """
    if not _is_package_available(package_name):
        install_cmd = (
            f"pip install {package_name}"
            if install_instruction is None
            else install_instruction
        )
        raise ImportError(
            f"The '{metric_name}' metric requires the `{package_name}` library but it was not found in your environment. "
            f"Please run: {install_cmd}"
        )


PROMPT_TEMPLATE = """You are a professional IETF RFC writer. 
Please revise the following text using your knowledge and understanding.

Input:

Original Text:
{old_text}

Output:

Revised Text:\n
"""

def avg_bert_score(bertscore_results: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate average BERTScore metrics.
    
    Args:
        bertscore_results: Dictionary containing precision, recall, and f1 scores
        
    Returns:
        Dictionary with averaged metrics
    """
    return {
        'avg_pre': np.mean(bertscore_results['precision']),
        'avg_rec': np.mean(bertscore_results['recall']),
        'avg_f1': np.mean(bertscore_results['f1'])
    }


class IdsTextFlow(Task):
    """Task for evaluating text revision quality using multiple metrics."""
    
    DEFAULT_DATASET_PATH = "../dataset/ids.i2c.test.generation.jsonl"
    DATASET_SPLIT = "train"  # The split name used by the dataset loader
    AVAILABLE_METRICS = ["bleu", "sacrebleu", "google_bleu", "bertscore", "meteor", "exact_match", "wer", "mauve"]
    
    def __init__(self, max_order: int = DEFAULT_MAX_ORDER, smooth: bool = DEFAULT_SMOOTH, dataset_path: str = None, metrics: List[str] = None):
        """Initialize IdsBase1 task.
        
        Args:
            max_order: Maximum n-gram order for BLEU score computation
            smooth: Whether to use smoothing for BLEU score
            dataset_path: Path to the dataset file (optional, defaults to DEFAULT_DATASET_PATH)
            metrics: List of metrics to compute (optional, defaults to all metrics)
        """
        # Set dataset path before calling parent init (parent needs it to load dataset)
        self.DATASET_PATH = dataset_path if dataset_path is not None else self.DEFAULT_DATASET_PATH
        self.metrics = metrics if metrics is not None else self.AVAILABLE_METRICS
        
        # Validate metrics
        invalid_metrics = [m for m in self.metrics if m not in self.AVAILABLE_METRICS]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {self.AVAILABLE_METRICS}")
        
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )
        self.max_order = max_order
        self.smooth = smooth
        logger.info(f"Initialized IdsBase1 with dataset: {self.DATASET_PATH}")
        logger.info(f"Enabled metrics: {self.metrics}")

    def get_dataset(self):
        """Returns dataset for the task.
        
        Note: Despite the 'train' split name, this actually loads the test data
        from DATASET_PATH. The split name is determined by how the dataset loader
        organizes the data, not the actual purpose of the data.
        
        Returns:
            Dataset containing test examples
        """
        return self.dataset[self.DATASET_SPLIT]

    def get_prompt(self, doc: Dict[str, str]) -> str:
        """Generate prompt for the given document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Formatted prompt string
        """
        return self.generate_prompt(doc["old_text"])

    def get_reference(self, doc: Dict[str, str]) -> str:
        """Builds the reference solution for the doc (sample from the test dataset).
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Reference text
        """
        return doc["new_text"]

    def get_old_text(self, doc: Dict[str, str]) -> str:
        """Extract old text from document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Original text
        """
        return doc["old_text"]

    def get_comments(self, doc: Dict[str, str]) -> str:
        """Extract comments from document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Comments text
        """
        return doc["comments"]

    def postprocess_generation(self, generation: str, idx: int) -> str:
        """Extract and clean the revised text from generation.
        
        Args:
            generation: Generated text from model
            idx: Index of the generation
            
        Returns:
            Cleaned output text
        """
        output = generation.split("Revised Text:\n", 1)[-1].strip()
        return output

    @staticmethod
    def generate_prompt(old_text: str) -> str:
        """Generate prompt from old text.
        
        Args:
            old_text: Original text to be revised
            
        Returns:
            Formatted prompt
        """
        return PROMPT_TEMPLATE.format(old_text=old_text)

    def _filter_empty_generations(
        self, 
        generations: List[str], 
        references: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter out empty generations and their corresponding references.
        
        Args:
            generations: List of generated texts
            references: List of reference texts
            
        Returns:
            Tuple of filtered (generations, references)
        """
        filtered_pairs = [
            (gen, ref) 
            for gen, ref in zip(generations, references) 
            if gen.strip()
        ]
        
        if not filtered_pairs:
            logger.warning("All generations are empty!")
            return [], []
        
        gens, refs = zip(*filtered_pairs)
        logger.info(f"Filtered {len(generations)} generations to {len(gens)} non-empty ones")
        return list(gens), list(refs)

    def process_results(
        self, 
        o_generations: List[List[str]], 
        o_references: List[str]
    ) -> Dict[str, Any]:
        """Process results and compute multiple evaluation metrics.
        
        Args:
            o_generations: List of generation lists (each containing one generation)
            o_references: List of reference texts
            
        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Loading evaluation metrics...")
        results = {}
        
        # Check and load metrics based on selection
        if "bleu" in self.metrics:
            requires_package("BLEU", "evaluate", "pip install evaluate")
            bleu = load("bleu")
        
        if "sacrebleu" in self.metrics:
            requires_package("SacreBLEU", "sacrebleu", "pip install sacrebleu")
            sacrebleu = load("sacrebleu")
        
        if "google_bleu" in self.metrics:
            requires_package("Google BLEU", "evaluate", "pip install evaluate")
            google_bleu = load("google_bleu")
        
        if "bertscore" in self.metrics:
            requires_package("BERTScore", "bert_score", "pip install bert-score")
            roberta_bertscore = load("bertscore")
            deberta_bertscore = load("bertscore", config_name="microsoft/deberta-xlarge-mnli")
        
        if "meteor" in self.metrics:
            requires_package("METEOR", "nltk", "pip install nltk")
            meteor = load('meteor')
        
        if "exact_match" in self.metrics:
            requires_package("Exact Match", "evaluate", "pip install evaluate")
            exact_match = load("exact_match")
        
        if "wer" in self.metrics:
            requires_package("WER", "jiwer", "pip install jiwer")
            wer = load("wer")
        
        if "mauve" in self.metrics:
            requires_package("MAUVE", "mauve", "pip install mauve-text")
            mauve = load('mauve')

        # Extract first generation from each list
        o_gens = [gen[0] for gen in o_generations]
        logger.info(f"Processing {len(o_gens)} generations")

        # Filter out empty strings and corresponding references
        gens, references = self._filter_empty_generations(o_gens, o_references)
        
        if not gens:
            logger.error("No valid generations to evaluate")
            return {}

        logger.info("Computing metrics...")
        
        # Compute only selected metrics
        if "bleu" in self.metrics:
            logger.info("Computing BLEU...")
            bleu_results = bleu.compute(
                references=references, 
                predictions=gens, 
                max_order=self.max_order, 
                smooth=self.smooth
            )
            results['bleu'] = bleu_results
        
        if "sacrebleu" in self.metrics:
            logger.info("Computing SacreBLEU...")
            sacre_bleu_results = sacrebleu.compute(
                references=references, 
                predictions=gens, 
                lowercase=True
            )
            results['sacre_bleu'] = sacre_bleu_results
        
        if "google_bleu" in self.metrics:
            logger.info("Computing Google BLEU...")
            google_bleu_results = google_bleu.compute(
                references=references, 
                predictions=gens
            )
            results['google_bleu'] = google_bleu_results
        
        if "bertscore" in self.metrics:
            logger.info("Computing BERTScore metrics...")
            roberta_bertscore_results = roberta_bertscore.compute(
                references=references, 
                predictions=gens, 
                lang="en"
            )
            avg_roberta_bertscore = avg_bert_score(roberta_bertscore_results)
            
            deberta_bertscore_results = deberta_bertscore.compute(
                references=references, 
                predictions=gens, 
                lang="en"
            )
            avg_deberta_bertscore = avg_bert_score(deberta_bertscore_results)
            
            results['roberta_bertscore'] = roberta_bertscore_results
            results['roberta_bertscore_avg'] = avg_roberta_bertscore
            results['deberta_bertscore'] = deberta_bertscore_results
            results['deberta_bertscore_avg'] = avg_deberta_bertscore
        
        if "meteor" in self.metrics:
            logger.info("Computing METEOR...")
            meteor_results = meteor.compute(
                references=references, 
                predictions=gens
            )
            results['meteor'] = meteor_results
        
        if "exact_match" in self.metrics:
            logger.info("Computing Exact Match...")
            exact_match_results = exact_match.compute(
                references=references, 
                predictions=gens, 
                ignore_case=True, 
                ignore_punctuation=True
            )
            results['EM'] = exact_match_results
        
        if "wer" in self.metrics:
            logger.info("Computing WER...")
            wer_results = wer.compute(
                references=references, 
                predictions=gens
            )
            results['wer'] = wer_results
        
        if "mauve" in self.metrics:
            logger.info("Computing MAUVE...")
            mauve_results = mauve.compute(
                references=references, 
                predictions=gens
            )
            results['mauve'] = mauve_results.mauve

        logger.info("Evaluation complete")
        return results
