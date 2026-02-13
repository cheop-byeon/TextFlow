import sys
import os

# Add parent directory to path so we can import ProtoSpec
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ProtoSpec.arguments import EvalArguments
from ProtoSpec.evaluator import Evaluator
from ProtoSpec.tasks import ALL_TASKS

import fnmatch
import json
import warnings
import logging
from typing import List, Dict, Any, Optional

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PRECISION_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class MultiChoice:
    """Helper class for handling multiple choice arguments with wildcard support."""
    
    def __init__(self, choices: List[str]):
        self.choices = choices

    def __contains__(self, values: str) -> bool:
        """Check if values match any choices (supports wildcards)."""
        return all(
            len(fnmatch.filter(self.choices, value)) > 0
            for value in values.split(",")
        )

    def __iter__(self):
        return iter(self.choices)


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=2048,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum length of extra generated text",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        default=None,
        help="Path to the dataset file for ids tasks (overrides default dataset paths)",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    return parser.parse_args()


def pattern_match(patterns: List[str], source_list: List[str]) -> List[str]:
    """Returns a list containing all values of the source_list that
    match at least one of the patterns.
    
    Args:
        patterns: List of patterns (supports wildcards)
        source_list: List of strings to match against
        
    Returns:
        List of matched strings
    """
    task_names = set()
    for pattern in patterns:
        task_names.update(fnmatch.filter(source_list, pattern))
    return sorted(task_names)


def get_gpus_max_memory(max_memory: str, num_gpus: int) -> Dict[int, str]:
    """Create a dictionary mapping GPU indices to memory limits.
    
    Args:
        max_memory: Memory limit per GPU
        num_gpus: Number of GPUs to use
        
    Returns:
        Dictionary mapping GPU index to memory limit
    """
    max_memory_dict = {i: max_memory for i in range(num_gpus)}
    logger.info(f"Loading model via {num_gpus} GPUs with max memory: {max_memory}")
    return max_memory_dict


def setup_logging():
    """Configure logging for transformers and datasets."""
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()


def get_task_names(args) -> List[str]:
    """Get list of tasks to evaluate."""
    if args.tasks is None:
        return ALL_TASKS
    return pattern_match(args.tasks.split(","), ALL_TASKS)


def main():
    """Main evaluation function."""
    args = parse_args()
    setup_logging()
    
    task_names = get_task_names(args)
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        logger.info(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        # Evaluation only mode - no code generation
        if accelerator.is_main_process:
            logger.info("Running in evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        # Generation mode - generate code and optionally evaluate
        if args.precision not in PRECISION_MAPPING:
            raise ValueError(
                f"Invalid precision '{args.precision}'. Choose from: {', '.join(PRECISION_MAPPING.keys())}"
            )

        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "token": args.use_auth_token,
        }
        
        if args.load_in_8bit:
            logger.info("Loading model in 8-bit quantization")
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": accelerator.process_index}
        elif args.load_in_4bit:
            logger.info("Loading model in 4-bit quantization")
            model_kwargs["load_in_4bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": accelerator.process_index}
        else:
            logger.info(f"Loading model in {args.precision} precision")
            model_kwargs["torch_dtype"] = PRECISION_MAPPING[args.precision]

            if args.max_memory_per_gpu:
                if args.max_memory_per_gpu == "auto":
                    model_kwargs["device_map"] = "auto"
                    logger.info("Loading model with automatic device mapping")
                else:
                    model_kwargs["max_memory"] = get_gpus_max_memory(
                        args.max_memory_per_gpu, accelerator.num_processes
                    )
                    model_kwargs["offload_folder"] = "offload"

        if args.modeltype == "causal":
            logger.info(f"Loading causal language model: {args.model}")
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        elif args.modeltype == "seq2seq":
            logger.warning(
                "Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models."
            )
            logger.info(f"Loading seq2seq model: {args.model}")
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs)
        else:
            raise ValueError(
                f"Invalid model type '{args.modeltype}'. Choose from: causal, seq2seq"
            )

        if args.peft_model:
            from peft import PeftModel  # Dynamic import to avoid dependency
            
            logger.info(f"Loading PEFT adapter from: {args.peft_model}")
            model = PeftModel.from_pretrained(model, args.peft_model)
            logger.info("Merging PEFT adapter with base model...")
            model.merge_and_unload()
            logger.info("PEFT merge complete")

        # Load tokenizer with appropriate padding configuration
        tokenizer_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "token": args.use_auth_token,
        }
        
        if args.left_padding:
            logger.info("Using left padding (required for some models like chatglm3-6b)")
            tokenizer_kwargs["padding_side"] = "left"
        else:
            tokenizer_kwargs["truncation_side"] = "left"
            tokenizer_kwargs["padding_side"] = "right"
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        
        # Configure special tokens
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                logger.info("Using bos_token as eos_token")
            else:
                raise ValueError("Tokenizer has neither eos_token nor bos_token")
        
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except AttributeError:
            logger.warning("Cannot set pad_token (read-only property)")

        evaluator = Evaluator(accelerator, model, tokenizer, args)

        for idx, task in enumerate(task_names):
            if args.generation_only:
                if accelerator.is_main_process:
                    logger.info(f"Generating code for task: {task}")
                generations, references, _, _ = evaluator.generate_text(task)
                if accelerator.is_main_process:
                    base_name = os.path.splitext(args.save_generations_path)[0]
                    save_generations_path = f"{base_name}_{task}.json"
                    save_references_path = f"references_{task}.json"
                    logger.info(f"Saving generations to: {save_generations_path}")
                    evaluator.save_json_files(
                        generations,
                        references,
                        save_generations_path,
                        save_references_path,
                    )
            else:
                if accelerator.is_main_process:
                    logger.info(f"Evaluating task: {task}")
                results[task] = evaluator.evaluate(task)

    # Save results and configuration
    results["config"] = vars(args)
    
    if not args.generation_only and accelerator.is_main_process:
        dumped = json.dumps(results, indent=2)
        logger.info("Evaluation Results:")
        print(dumped)
        
        logger.info(f"Saving results to: {args.metric_output_path}")
        with open(args.metric_output_path, "w") as f:
            f.write(dumped)
        logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
