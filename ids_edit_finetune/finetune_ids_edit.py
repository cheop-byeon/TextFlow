import logging
import argparse
import os
from pathlib import Path
from huggingface_hub import login
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from accelerate import Accelerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama model using LoRA")
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="./base_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Path to the model or model ID from Hugging Face"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".ids/train/i2c/ids.i2c.train.generation.jsonl",
        help="Path to the training data file"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./outputs",
        help="Base directory for saving outputs (will create subdirectory based on model name)"
    )
    
    # Optional training parameters
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Configuration
MAX_SEQ_LENGTH = args.max_seq_length
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
WARMUP_STEPS = args.warmup_steps
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
MODEL_NAME = args.model_name
DATA_PATH = args.data_path

# Create output directories based on model name
model_identifier = Path(MODEL_NAME).name.lower().replace("-", "_")
OUTPUT_DIR = os.path.join(args.output_base_dir, f"{model_identifier}_training")
MODEL_SAVE_NAME = os.path.join(args.output_base_dir, f"{model_identifier}_fine_tuned")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_NAME, exist_ok=True)

logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Data: {DATA_PATH}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Model save path: {MODEL_SAVE_NAME}")

PROMPT_TEMPLATE = """You are a professional IETF RFC writer. 
### Instruction:
Identify the parts of the original text that need revision based on the feedback. Revise the text accordingly.

### Input:

Original Text:
{}

Feedback:
{}

### Output:

Revised Text:
{}
"""

# Initialize model and tokenizer
logger.info("Loading model and tokenizer...")
dtype = None
accelerator = Accelerator()
num_gpus = accelerator.num_processes
fastLM, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=dtype
)
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """Format examples into prompt template with EOS token."""
    inputs = examples["old_text"]
    reviews = examples["comments"]
    outputs = examples["new_text"]
    
    texts = [
        PROMPT_TEMPLATE.format(i, r, o) + EOS_TOKEN
        for i, r, o in zip(inputs, reviews, outputs)
    ]
    return {"text": texts}


def main():
    """Main training function."""
    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_dataset(
            'json',
            data_files=DATA_PATH,
            split="train",
            trust_remote_code=True
        )
        
        logger.info(f"Dataset size: {len(dataset)} examples")
        
        # Format dataset with batching for efficiency
        logger.info("Formatting dataset...")
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            desc="Formatting prompts",
            num_proc=num_gpus
        )
        
        logger.info(f"Sample text:\n{dataset['text'][0]}")

        # Configure LoRA parameters
        logger.info("Configuring LoRA model...")
        peft_config = {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 16,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407,
            "use_rslora": False,
            "loftq_config": None,
        }
        
        peft_model = FastLanguageModel.get_peft_model(fastLM, **peft_config)

        # Configure training
        logger.info("Initializing trainer...")
        bf16_support = is_bfloat16_supported()
        training_args = TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not bf16_support,
            bf16=bf16_support,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to=[],
            save_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=num_gpus,
            args=training_args,
        )

        # Train
        logger.info("Starting training...")
        trainer_stats = trainer.train()

        # Log training statistics
        train_time = trainer_stats.metrics['train_runtime']
        logger.info(f"Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
        
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(f"Peak reserved memory: {used_memory} GB")

        # Save model
        logger.info(f"Saving model to {MODEL_SAVE_NAME}...")
        peft_model.save_pretrained(MODEL_SAVE_NAME)
        tokenizer.save_pretrained(MODEL_SAVE_NAME)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
