# Using Custom Dataset Paths with IDS Tasks

All `ids_*` task scripts now support parameterized dataset paths, allowing you to override the default dataset location without modifying the code.

## Overview
We recommend you to download the dataset from huggingface and store them in a local folder:

https://huggingface.co/datasets/jiebi/TextFlow/

Each task has a default dataset path:
- `dev_data`: `./dataset/ids.i2c.dev.generation.jsonl`
- `test_data`: `./dataset/ids.i2c.test.generation.jsonl`
- `test_data_supp`: `./dataset/ids.i2c.supplementary.generation.jsonl`

## Downloading Datasets

### Quick Start

Download all datasets to the `dataset/` directory:

```bash
python download_dataset.py
```

### Download Specific Split

Download only the test dataset:

```bash
python download_dataset.py --split test
```

Download only the dev dataset:

```bash
python download_dataset.py --split dev
```

Download only the supplementary dataset:

```bash
python download_dataset.py --split supp
```

### Verify Datasets

Check if all datasets are present and display their sizes:

```bash
python download_dataset.py --verify-only
```

### Requirements

Make sure you have the `datasets` package installed:

```bash
pip install datasets huggingface-hub
```

If you're downloading private datasets, authenticate with HuggingFace:

```bash
huggingface-cli login
```

### Via Command Line

Pass the `--load_dataset_path` argument to override the default:

```bash
python main.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --tasks ids_edit \
    --load_dataset_path "./dataset/ids.i2c.test.generation.jsonl" \
    --batch_size 4 \
    --max_new_tokens 512
```

### Multiple Tasks with Same Dataset

```bash
python main.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --tasks ids_auto_complete,ids_startup,ids_edit \
    --load_dataset_path "/path/to/custom/dataset.jsonl" \
    --batch_size 4
```

### Using Default Paths

If `--load_dataset_path` is not provided, each task uses its default:

```bash
python main.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --tasks ids_edit \
    --batch_size 4
```

## Programmatic Usage

You can also instantiate tasks directly with custom paths:

```python
from ProtoSpec.tasks import ids_edit

# Use custom dataset path
task = ids_edit.IdsTextFlow(
    dataset_path="/custom/path/to/dataset.jsonl"
)

# Use default path
task_default = ids_edit.IdsTextFlow()
```

## Benefits

1. **No Code Modifications**: Switch between local and cluster datasets via command line
2. **Consistent Interface**: All ids tasks support the same parameter
3. **Backward Compatible**: Existing scripts work without changes
4. **Logging**: Each task logs which dataset path it's using for debugging

## Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=ids_eval
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

MODEL_NAME="meta-llama/Meta-Llama-3-8B"
DATASET_PATH="./dataset/ids.i2c.test.generation.jsonl"

python main.py \
    --model "$MODEL_NAME" \
    --tasks ids_edit \
    --load_dataset_path "$DATASET_PATH" \
    --batch_size 4 \
    --max_new_tokens 512 \
    --save_generations \
    --metric_output_path "results_${SLURM_JOB_ID}.json"
```
