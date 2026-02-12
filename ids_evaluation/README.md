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
    --load_dataset_path ./ids/ids.i2c.test.generation.jsonl \
    --save_generations \
    --save_generations_path generations.json \
    --metric_output_path metrics.json \
    --seed 42
```

## Acknowledgements

This evaluation harness is derived from the [BigCode evaluation harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
