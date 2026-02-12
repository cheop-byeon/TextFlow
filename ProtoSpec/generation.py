import json
import logging
from math import ceil
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList

from ProtoSpec.utils import TokenizedDataset, complete_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom StoppingCriteria which checks if all generated functions in the batch are completed."""
    
    def __init__(self, start_length: int, eof_strings: List[str], tokenizer, check_fn=None):
        """Initialize EndOfFunctionCriteria.
        
        Args:
            start_length: Starting length to trim from generated sequences
            eof_strings: List of end-of-function strings to check for
            tokenizer: Tokenizer to decode generated sequences
            check_fn: Optional custom check function
        """
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                stop_string in decoded_generation for stop_string in self.eof_strings
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Check if all generated sequences contain any of the end-of-function strings.
        
        Args:
            input_ids: Generated token IDs
            scores: Generation scores
            
        Returns:
            True if all sequences are complete
        """
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        return all(self.check_fn(decoded_generation) for decoded_generation in decoded_generations)


class TooLongFunctionCriteria(StoppingCriteria):
    """Custom StoppingCriteria which checks if the generated function is too long 
    by a certain multiplier based on input length."""

    def __init__(self, input_length: int, multiplier: float):
        """Initialize TooLongFunctionCriteria.
        
        Args:
            input_length: Original input length
            multiplier: Maximum length multiplier
        """
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Check if generated sequence is too long.
        
        Args:
            input_ids: Generated token IDs
            scores: Generation scores
            
        Returns:
            True if sequence exceeds maximum length
        """
        return input_ids.shape[1] > int(self.input_length * self.multiplier)


def _setup_generation_config(args, generation_config) -> dict:
    """Setup generation configuration parameters.
    
    Args:
        args: Arguments containing generation settings
        generation_config: Default generation configuration
        
    Returns:
        Dictionary of generation kwargs
    """
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.temperature else generation_config.temperature,
        "top_p": args.top_p if args.top_p else generation_config.top_p,
        "top_k": args.top_k if args.top_k is not None else generation_config.top_k,
        "repetition_penalty": generation_config.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }
    logger.info(f"Generation config: {gen_kwargs}")
    return gen_kwargs


def _setup_stopping_criteria(task, tokenizer, args) -> Optional[StoppingCriteriaList]:
    """Setup stopping criteria for generation.
    
    Args:
        task: Task instance
        tokenizer: Tokenizer instance
        args: Arguments containing instruction tokens
        
    Returns:
        StoppingCriteriaList or None
    """
    stopping_criteria = []
    
    # Add EOS token to stop words if available
    if task.stop_words and tokenizer.eos_token:
        task.stop_words.append(tokenizer.eos_token)
    
    # Setup end-of-function criteria
    if hasattr(task, "check_fn"):
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
        )
    elif task.stop_words:
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer)
        )
    
    # Setup length-based criteria
    if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
        stopping_criteria.append(
            TooLongFunctionCriteria(0, task.max_length_multiplier)
        )
    
    if stopping_criteria:
        logger.info(f"Configured {len(stopping_criteria)} stopping criteria")
        return StoppingCriteriaList(stopping_criteria)
    
    return None


def _prepare_model_and_dataloader(model, ds_loader, accelerator, args):
    """Prepare model and dataloader for generation.
    
    Args:
        model: Model to prepare
        ds_loader: DataLoader to prepare
        accelerator: Accelerator instance
        args: Arguments containing model configuration
        
    Returns:
        Tuple of (prepared_model, prepared_dataloader, is_wrapped)
    """
    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    is_wrapped = is_loaded_in_8bit or is_loaded_in_4bit
    
    if args.max_memory_per_gpu is not None:
        # Model is already sharded across multiple GPUs
        logger.info("Using pre-sharded model across GPUs")
        ds_loader = accelerator.prepare(ds_loader)
    elif not is_wrapped:
        # Wrap data loader only to avoid extra memory occupation
        logger.info("Moving model to accelerator device")
        model = model.to(accelerator.device)
        ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        logger.info("Preparing 8bit/4bit model with accelerator")
        model, ds_loader = accelerator.prepare(model, ds_loader)
    
    return model, ds_loader, is_wrapped


def parallel_generations(
        task,
        dataset,
        accelerator,
        model,
        tokenizer,
        n_tasks: int,
        args,
        curr_sample_idx: int = 0,
) -> List[List[str]]:
    """Generate code in parallel across multiple devices.
    
    Args:
        task: Task instance
        dataset: Dataset to generate from
        accelerator: Accelerator for distributed generation
        model: Language model
        tokenizer: Tokenizer
        n_tasks: Number of tasks to generate
        args: Arguments containing generation configuration
        curr_sample_idx: Current sample index to start from
        save_every_k_tasks: Save intermediate results every k tasks (-1 to disable)
        intermediate_generations: Existing intermediate generations to continue from
        intermediate_save_generations_path: Path to save intermediate generations
        
    Returns:
        List of generated code samples
    """
    # Load pre-generated code if specified
    if args.load_generations_path:
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
        if accelerator.is_main_process:
            logger.info(
                f"Loaded generations: {n_tasks} selected from {len(generations)} "
                f"with {len(generations[0])} candidates"
            )
        return generations[:n_tasks]

    # Setup random seed
    set_seed(args.seed, device_specific=True)
    logger.info(f"Set random seed to {args.seed}")
    
    # Load generation configuration
    generation_config = GenerationConfig.from_pretrained(args.model)
    gen_kwargs = _setup_generation_config(args, generation_config)
    
    # Setup stopping criteria
    stopping_criteria_list = _setup_stopping_criteria(task, tokenizer, args)
    if stopping_criteria_list:
        gen_kwargs["stopping_criteria"] = stopping_criteria_list
    
    # Log task information
    if accelerator.is_main_process:
        logger.info(f"Processing {n_tasks} problems for this task")
    
    # Calculate number of copies needed
    n_copies = ceil(args.n_samples / args.batch_size)
    logger.info(f"Generating {n_copies} copies per sample (n_samples={args.n_samples}, batch_size={args.batch_size})")
    
    # Create tokenized dataset
    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_generation,
        limit_start=args.limit_start + curr_sample_idx,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
    )

    # Create dataloader (batch_size=1 as batching is handled internally)
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    # Prepare model and dataloader
    model, ds_loader, _ = _prepare_model_and_dataloader(
        model, ds_loader, accelerator, args
    )

    # Generate text completions
    logger.info("Starting text generation...")
    generations = complete_text(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        limit_start=args.limit_start + curr_sample_idx,
        batch_size=args.batch_size,
        prefix=args.prefix,
        postprocess=args.postprocess,
        **gen_kwargs,
    )
    
    logger.info(f"Code generation complete: {len(generations)} samples generated")
    return generations
