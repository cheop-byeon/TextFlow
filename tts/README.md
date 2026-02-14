# Text Generation with Think-and-Wait

This directory contains scripts for generating text revisions using Qwen models with optional thinking and waiting phases.

## Main Script

### `my_gen_qwen_32b_think_and_wait.py`

Generate text with optional thinking and waiting phases using the Qwen 2.5 32B Instruct model.

**Key Features:**
- Optional "wait" phase after the thinking phase
- Configurable model and dataset paths
- Outputs separate files for thinking process and final revisions

**Command-Line Arguments:**

- `--use_wait` - Enable the wait phase after thinking (default: `False`)
- `--model_path` - Path to the model (default: `../Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`)
- `--dataset_path` - Path to the dataset (default: `../dataset/ids.i2c.test.generation.jsonl`)

**Usage Examples:**

```bash
# Without wait phase (default)
python ids_edit_reasoning_tts.py

# With wait phase
python ids_edit_reasoning_tts.py --use_wait

# Custom paths
python ids_edit_reasoning_tts.py --use_wait \
    --model_path ./path/to/model \
    --dataset_path ./path/to/dataset.jsonl
```

**Output Files:**

The script generates two types of output files:

1. **Thinking Process:** `think_qwen_32b_{with_wait|no_wait}.json`
   - Contains the model's thinking/reasoning process
   
2. **Final Revisions:** `revised_qwen_32b_{with_wait|no_wait}.json`
   - Contains the final revised text output

The filename suffix (`_with_wait` or `_no_wait`) indicates whether the wait phase was enabled.

## How It Works

1. **Think Phase:** The model first thinks about the revision task
2. **Wait Phase (Optional):** If enabled, the model continues thinking with a "Wait" prompt
3. **Final Answer:** The model generates the final revised text

The optional wait phase allows the model more time to consider complex feedback before producing the final output.
