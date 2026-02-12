#!/bin/bash

# ==============================================================================
# Fine-tune Llama Model using LoRA with Unsloth
# ==============================================================================
#
# USAGE:
#   Edit the configuration variables below, then run: ./finetune_llama.sh
#
# OUTPUT:
#   - Training checkpoints: outputs/<model_name>_training/
#   - Fine-tuned model: outputs/<model_name>_fine_tuned/
#
# ==============================================================================

#SBATCH --job-name=ft
#SBATCH --account=12345678
#SBATCH --time=00-05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=a100
#SBATCH --gpus=1

# ==============================================================================
# Configuration - Edit these variables as needed
# ==============================================================================

# Model configuration
MODEL_NAME="./Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DATA_PATH="./dataset/i2c/ids.i2c.train.generation.jsonl"
OUTPUT_BASE_DIR="./outputs"

# Training parameters
MAX_SEQ_LENGTH=2048
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
WARMUP_STEPS=5
NUM_EPOCHS=5
LEARNING_RATE=2e-4

# ==============================================================================
# Launch training
# ==============================================================================

accelerate launch finetune_ids_edit.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE
