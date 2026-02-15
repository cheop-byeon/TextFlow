# TextFlow Tasks Evaluation - Parameters Reference

This document describes the parameters and functions available for the TextFlow evaluation harness.

## Input Parameters (main.py)

### Model and Loading
* `--model`: HuggingFace model repo or local path.
* `--modeltype`: `causal` or `seq2seq` for AutoModel selection.
* `--peft_model`: Optional PEFT adapter (e.g., LoRA). Requires `--model` to be the base model.
* `--revision`: Model revision or tag to load.
* `--use_auth_token`: Use your Hugging Face token (required for private models).
* `--trust_remote_code`: Allow custom model code from the Hub (use only if you trust the source).

### Tasks and Prompting
* `--tasks`: Comma-separated task names; supports wildcards (must match `ALL_TASKS`).
  - Available tasks: `ids_auto_complete`, `ids_startup`, `ids_edit`

### Generation and Decoding
* `--batch_size`: Per-worker batch size (default: 1).
* `--max_length_generation`: Max total length (prompt + generation) in tokens (default: 2048).
* `--max_new_tokens`: Max length of newly generated tokens (default: 512).
* `--precision`: `fp32`, `fp16`, or `bf16` (default: `fp32`).
* `--load_in_8bit`: Load model in 8-bit quantization.
* `--load_in_4bit`: Load model in 4-bit quantization.
* `--left_padding`: Force left padding (needed for some models like chatglm3-6b).
* `--prefix`: String prefix to add before all prompts (default: "").

### Dataset and Limits
* `--limit`: Number of samples to evaluate (default: None = all).
* `--limit_start`: Starting offset when limiting samples (default: 0).
* `--load_dataset_path`: Custom dataset path for ids tasks (overrides defaults).
  - Format: JSONL file with the task-specific fields (old_text, comments, new_text)

### Evaluation Flow
* `--postprocess`: Enable/disable postprocessing (default: True).
* `--generation_only`: Generate only, skip evaluation.
* `--load_generations_path`: Evaluate previously generated solutions (skip generation).
* `--check_references`: Validate reference solutions without generation.

### Outputs
* `--metric_output_path`: Output JSON file for metrics (default: `evaluation_results.json`).
* `--save_generations`: Save generated text to file.
* `--save_generations_path`: Path for saving generations (default: `generations.json`).
* `--save_references`: Save reference solutions alongside generations.
* `--save_references_path`: Path for saving references (default: `references.json`).

### Reproducibility
* `--seed`: Random seed for generation.

### Memory Management
* `--max_memory_per_gpu`: Maximum memory per GPU (e.g., "20GB" or "auto").

## Notes on Parameters

* `max_length_generation` is the maximum token length including the input prompt. For long prompts, increase this value.
* `batch_size` should be less than or equal to `n_samples` when sampling multiple candidates.
* Some models with custom code on the HF hub require `--trust_remote_code`; for private models add `--use_auth_token`.
* The `--load_dataset_path` parameter allows you to override the default dataset paths for evaluation tasks.

## Example Command

```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks ids_edit \
    --batch_size 1 \
    --max_new_tokens 512 \
    --load_dataset_path ../dataset/ids.i2c.test.generation.jsonl \
    --save_generations \
    --save_generations_path generations.json \
    --metric_output_path metrics.json \
    --seed 42
```

## Task Compatibility

All three tasks support the same metrics:

- `ids_auto_complete`: Text completion task
- `ids_edit`: Text revision with feedback task
- `ids_startup`: Feedback-only prompt task

You can specify different metrics for different runs to compare evaluation approaches.

# Evaluation Metrics Usage Guide

## Overview

The evaluation script now supports selective metric computation through the `--metrics` parameter. This allows you to choose which evaluation metrics to compute, avoiding unnecessary package installations and speeding up evaluation.

## Available Metrics

- `bleu`: BLEU score (requires: `evaluate`)
- `sacrebleu`: SacreBLEU score (requires: `sacrebleu`)
- `google_bleu`: Google BLEU score (requires: `evaluate`)
- `bertscore`: BERTScore with RoBERTa and DeBERTa models (requires: `bert-score`)
- `meteor`: METEOR score (requires: `nltk`)
- `exact_match`: Exact Match score (requires: `evaluate`)
- `wer`: Word Error Rate (requires: `jiwer`)
- `mauve`: MAUVE score (requires: `mauve-text`)

## Usage

### Compute All Metrics (Default)

If you don't specify `--metrics`, all available metrics will be computed:

```bash
python main.py \
    --model_path <model_path> \
    --tasks ids_auto_complete \
    --output_path results.json
```

### Compute Selected Metrics

Specify one or more metrics to compute:

```bash
# Compute only BLEU and SacreBLEU
python main.py \
    --model_path <model_path> \
    --tasks ids_auto_complete \
    --metrics bleu sacrebleu \
    --output_path results.json

# Compute only BERTScore
python main.py \
    --model_path <model_path> \
    --tasks ids_edit \
    --metrics bertscore \
    --output_path results.json

# Compute multiple metrics
python main.py \
    --model_path <model_path> \
    --tasks ids_startup \
    --metrics bleu sacrebleu meteor bertscore \
    --output_path results.json
```

## Package Requirements

Each metric requires specific packages. If a selected metric's package is not installed, you'll get a clear error message with installation instructions:

```
ImportError: The 'MAUVE' metric requires the `mauve` library but it was not found in your environment. 
Please run: pip install mauve-text
```

### Minimal Installation

For fastest setup, install only the packages for metrics you need:

```bash
# For BLEU, Google BLEU, Exact Match only
pip install evaluate

# Add SacreBLEU
pip install sacrebleu

# Add BERTScore
pip install bert-score

# Add METEOR
pip install nltk

# Add WER
pip install jiwer

# Add MAUVE
pip install mauve-text
```

### Full Installation

To use all metrics:

```bash
pip install evaluate sacrebleu bert-score nltk jiwer mauve-text
```

## Performance Benefits

Selecting specific metrics provides several benefits:

1. **Faster Installation**: Only install packages you need
2. **Faster Evaluation**: Skip computation of unnecessary metrics
3. **Reduced Memory**: Some metrics (especially BERTScore and MAUVE) require large models
4. **Cleaner Results**: Output only contains the metrics you care about

## Examples

### Quick BLEU-only Evaluation

```bash
python ids_evaluation/main.py \
    --model_path meta-llama/Llama-3.2-8B-Instruct \
    --tasks ids_auto_complete \
    --metrics bleu \
    --output_path quick_results.json
```

### Comprehensive Semantic Evaluation

```bash
python ids_evaluation/main.py \
    --model_path meta-llama/Llama-3.2-8B-Instruct \
    --tasks ids_edit \
    --metrics bertscore meteor mauve \
    --output_path semantic_results.json
```

### Standard Metrics Suite

```bash
python ids_evaluation/main.py \
    --model_path meta-llama/Llama-3.2-8B-Instruct \
    --tasks ids_startup \
    --metrics bleu sacrebleu meteor bertscore \
    --output_path standard_results.json
```
