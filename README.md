# TextFlow Tasks

TextFlow is an evaluation framework for internet-drafts/RFCs text revision tasks. It provides three main task types for evaluating different text generation scenarios.

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/cheop-byeon/TextFlow.git
cd TextFlow

# Install PyTorch (choose based on your hardware: https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install TextFlow and dependencies
pip install -e .

# Configure distributed training (optional, for multi-GPU)
accelerate config

# Login to HuggingFace Hub (for private models)
huggingface-cli login
```

## Task Overview

### 1. **Autocomplete** (`ids_auto_complete`)
Generates text completions based on the original text.

**Prompt Template:**
```
You are a professional IETF RFC writer. 
Please revise the following text using your knowledge and understanding.

Input:
Original Text:
{old_text}

Output:
Revised Text:
```

**Use Case:** Evaluating the model's ability to continue and improve incomplete or draft text.

---

### 2. **Startup** (`ids_startup`)
Generates revised text based solely on feedback, without seeing the original text.

**Prompt Template:**
```
You are a professional IETF RFC writer. 
Below is some feedback discussing changes needed for a text. 
Please provide a revised version of the text based solely on the feedback.

Input:
Feedback:
{feedback}

Output:
Revised Text:
```

**Use Case:** Evaluating the model's ability to perform blind revisions using only feedback as guidance.

---

### 3. **Edit** (`ids_edit`)
Generates revised text by combining both the original text and feedback.

**Prompt Template:**
```
You are a professional IETF RFC writer. 
Identify the parts of the original text that need revision based on the feedback.
Revise the text accordingly.

Input:
Original Text:
{old_text}

Feedback:
{feedback}

Output:
Revised Text:
```

**Use Case:** Evaluating the model's ability to perform targeted revisions using both context and feedback.

---

## Task Structure

All tasks inherit from the `Task` base class and implement:
- `get_dataset()` - Load the evaluation dataset
- `get_prompt()` - Generate the prompt for a sample
- `get_reference()` - Get the reference/gold standard revision
- `postprocess_generation()` - Clean up model output
- `process_results()` - Aggregate evaluation metrics

## Evaluation Metrics

Tasks evaluate text revisions using multiple metrics:
- **BLEU** - Lexical overlap with reference
- **SacreBLEU** - Corpus-level BLEU score
- **Google BLEU** - N-gram based similarity
- **BERTScore** - Semantic similarity (RoBERTa and DeBERTa variants)
- **METEOR** - Alignment-based metric
- **Exact Match** - Perfect match with reference
- **WER** - Word Error Rate
- **MAUVE** - Distribution distance metric

## Running Tasks

### Basic Usage

Navigate to the evaluation directory and run the main script:

```bash
cd ids_evaluation
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks ids_edit \
    --batch_size 1 \
    --max_new_tokens 512 \
    --load_dataset_path ../ids/ids.i2c.test.generation.jsonl \
    --save_generations \
    --metric_output_path metrics.json
```

### Generation and Evaluation Modes

**Generation Only:**
```bash
python main.py --model <model_id> --tasks <task> --generation_only --save_generations
```

**Evaluation Only (with pre-generated outputs):**
```bash
python main.py --model <model_id> --tasks <task> --load_generations_path <path_to_generations.json>
```

### Using PEFT Adapters

```bash
python main.py \
    --model <base_model_id> \
    --peft_model <path_to_peft_adapter> \
    --tasks ids_edit \
    --batch_size 1
```

### Quantization Support

**4-bit quantization:**
```bash
python main.py --model <model_id> --load_in_4bit --tasks ids_edit
```

**8-bit quantization:**
```bash
python main.py --model <model_id> --load_in_8bit --tasks ids_edit
```

### Multi-GPU Evaluation

```bash
accelerate launch main.py \
    --model <model_id> \
    --tasks <task> \
    --batch_size 2
```

## Parameter Reference

For detailed parameter documentation, see [ids_evaluation/README.md](ids_evaluation/README.md).

### Key Parameters

- `--model` - HuggingFace model ID or local path (required)
- `--tasks` - Task names: `ids_auto_complete`, `ids_startup`, `ids_edit`
- `--batch_size` - Batch size per GPU (default: 1)
- `--max_new_tokens` - Maximum new tokens to generate (default: 512)
- `--load_dataset_path` - Path to custom dataset (JSONL format)
- `--seed` - Random seed for reproducibility

## Dataset Format

Datasets should be in JSONL format with the following structure:

**For ids_auto_complete:**
```json
{"old_text": "original text content"}
```

**For ids_startup:**
```json
{"comments": "feedback or review comments"}
```

**For ids_edit:**
```json
{"old_text": "original text content", "comments": "feedback or review comments"}
```

---

## Implementation Details

- **Framework:** Built with HuggingFace Transformers and Accelerate
- **Support:** Causal LM (GPT-style) and Seq2Seq models
- **Quantization:** 4-bit and 8-bit support via bitsandbytes
- **Distributed:** Multi-GPU evaluation via Accelerate

---

## Quick Start

1. Prepare your evaluation dataset in JSONL format
2. Choose a task type based on your evaluation scenario
3. Select a model from HuggingFace Hub
4. Run the evaluation command with desired parameters
5. Results will be saved as JSON with all metrics
