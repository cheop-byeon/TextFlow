#!/bin/bash
# TextFlow IDS Edit Task Evaluation Script
# 
# This script evaluates the Llama 3.1 8B Instruct model (via Unsloth) on the IDS Edit task.
# The IDS Edit task involves generating revised text based on both original text and feedback.
#
# SLURM Configuration:
# - Runs on a single GPU
# - Estimated runtime: ~1 hour
# - Single task per node for isolation
#
# Dataset:
# - Input: ids.i2c.test.generation.jsonl (test set)
# - Format: JSONL with old_text, comments, and new_text fields
#
# Generation Parameters:
# - Model: Meta-Llama-3.1-8B-Instruct (4-bit quantized via Unsloth)
# - Temperature: 0.1 (low entropy for focused generation)
# - Samples: 1 generation per prompt
# - Max tokens: 512
# - Batch size: 1
#
# Outputs:
# - generations: unsloth_llama_8b_ids_edit.json (model-generated revisions)
# - metrics: unsloth_llama_8b_ids_edit_metric.json (BLEU, BERTScore, METEOR, etc.)
#
# If using custom model, use peft_model parameter to specify the path to the PEFT model.
#
# Post-generation:
# Use --load_generations_path to re-evaluate without regenerating (faster evaluation runs)

#SBATCH --job-name=ids_edit
#SBATCH --account=12345677
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=partition_name
#SBATCH --gpus=1


accelerate launch  main.py \
--model ../unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
--tasks ids_edit \
--do_sample True \
--temperature 0.1 \
--n_samples 1 \
--batch_size 1 \
--save_generations \
--trust_remote_code \
--load_dataset_path ../dataset/ids.i2c.test.generation.jsonl \
--save_generations_path unsloth_llama_8b_ids_edit.json \
--metric_output_path unsloth_llama_8b_ids_edit_metric.json \
--max_new_tokens 512 \
--seed 42

# Post-evaluation example:
# To re-evaluate the generated outputs without regenerating, use:
# python main.py \
#     --tasks ids_edit \
#     --load_generations_path unsloth_llama_8b_ids_edit.json \
#     --metric_output_path unsloth_llama_8b_ids_edit_metric.json
