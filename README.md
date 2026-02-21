# TextFlow: RFC Text Revision Evaluation Framework

**A comprehensive evaluation framework for assessing large language models' ability to revise Internet-Draft (RFC) documents through three complementary text editing scenarios.**

## Overview

TextFlow provides a reproducible research platform for evaluating text revision capabilities in the context of Internet Engineering Task Force (IETF) RFC documents. The framework implements three distinct evaluation tasks that test different aspects of LLM writing assistance:

- **Autocomplete**: Improving incomplete/draft text (zero-context revision)
- **Startup**: Revising text from feedback alone (blind revision)
- **Edit**: Revising text using both original and feedback (context-aware revision)

This repository contains the implementation and evaluation pipeline used in the following publications:

### Slides

Conference presentations are available in the [slides](slides/) directory:

- **NLDB 2025** - [slides/NLDB2025.pdf](slides/NLDB2025.pdf)
  - International Conference on Applications of Natural Language to Information Systems
  - Program: https://www.jaist.ac.jp/event/nldb2025/program.html

- **ANRW 2025** - [slides/ANRW2025.pdf](slides/ANRW2025.pdf)
  - ACM/IRTF Applied Networking Research Workshop
  - Program: https://www.irtf.org/anrw/2025/program.html

- **IETF 124 Montreal** - [slides/ietf124Montreal.pdf](slides/ietf124Montreal.pdf)
  - Research and Analysis of Standard-Setting Processes Research Group Session
  - Program: https://datatracker.ietf.org/meeting/124/session/rasprg

### Videos

- **IETF 123 - Applied Networking Research Workshop (ANRW)**
  - Date: July 22, 2025 at 09:30 UTC
  - Video: https://www.youtube.com/watch?v=45Y_IwJi9y0
  - Session: Research presentations on LLM-enhanced RFC writing and IETF collaboration tools

- **IETF 124 - RASPRG Working Group**
  - Date: November 5, 2025 at 15:30-17:30 CET
  - Video: https://www.youtube.com/watch?v=rlYnQ5V8B64
  - Session: Research and Analysis of Standard-Setting Processes Research Group

### Related Publications

```
@inproceedings{bian2025instruction,
  title={Instruction Tuning TextFlow Semi-automatic RFCs Generation},
  author={Bian, Jie and Welzl, Michael},
  booktitle={International Conference on Applications of Natural Language to Information Systems},
  pages={350--364},
  year={2025}
}

@inproceedings{bian2025empowering,
  title={Empowering IETF Collaboration with NLP Search Innovations and LLM-Enhanced RFC Writing},
  author={Bian, Jie and Welzl, Michael},
  booktitle={Proceedings of the 2025 Applied Networking Research Workshop},
  pages={24--31},
  year={2025}
}
```

---

## Table of Contents

- [Installation](#installation)
- [Evaluation Tasks](#evaluation-tasks)
- [Dataset Format](#dataset-format)
- [Quick Start](#quick-start)
- [Running Evaluations](#running-evaluations)
- [Evaluation Metrics](#evaluation-metrics)
- [Advanced Features](#advanced-features)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Installation

### System Requirements

- **Python**: 3.11+
- **PyTorch**: 2.9.1 with CUDA support (recommended)
- **GPU**: NVIDIA GPU with CUDA 12.6+ support (for inference acceleration)
- **Disk Space**: ~20GB (model weights + datasets)

### Dependencies

Clone and install the repository:

```bash
git clone https://github.com/cheop-byeon/TextFlow.git
cd TextFlow
pip install -e .
```

Or install with conda (recommended for managing CUDA dependencies):

```bash
conda create -p path/to/conda_env python=3.12
conda activate path/to/conda_env
```

### HuggingFace Hub Configuration

For accessing private models or downloading large model weights:

```bash
huggingface-cli login
```

### Distributed Training Setup (Optional)

For multi-GPU evaluation:

```bash
accelerate config
```

---

## Evaluation Tasks

### 1. Autocomplete Task (`ids_auto_complete`)

**Objective**: Evaluate the model's ability to improve incomplete or draft RFC text.

**Evaluation Scenario**: Zero-context revision where the model must enhance text quality without external feedback.

**Input**: Original text passage from an RFC document
**Output**: Revised/improved version of the text

**Prompt Template**:
```
You are a professional IETF RFC writer. 
Please revise the following text using your knowledge and understanding.

Input:
Original Text:
{old_text}

Output:
Revised Text:
```

**Assessment Basis**: Lexical and semantic similarity between model-generated revisions and human reference revisions.

---

### 2. Startup Task (`ids_startup`)

**Objective**: Evaluate the model's ability to revise text based exclusively on feedback guidance (blind revision).

**Evaluation Scenario**: Feedback-driven revision where the model must generate improved text without seeing the original.

**Input**: Feedback or review comments describing needed changes
**Output**: Revised text that incorporates the feedback suggestions

**Prompt Template**:
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

**Assessment Basis**: Model's ability to infer and apply improvements from textual guidance alone.

---

### 3. Edit Task (`ids_edit`)

**Objective**: Evaluate context-aware text revision combining original text and feedback (standard editing scenario).

**Evaluation Scenario**: Traditional document editing where the model has both source material and revision guidance.

**Input**: Original text + Feedback/review comments
**Output**: Revised text incorporating both context and feedback

**Prompt Template**:
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
    --load_dataset_path ../dataset/ids.i2c.test.generation.jsonl \
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


## Acknowledgements

This evaluation harness is derived from the [BigCode evaluation harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
