# Text Generation with Think-and-Wait

This directory contains scripts for generating text revisions using Qwen models with optional thinking and waiting phases.

## Main Script

### `ids_edit_reasoning_tts.py`

Generate text with optional thinking and waiting phases using the reasoning models, e.g., Qwen 2.5 32B Instruct model.

**Key Features:**
- Optional "wait" phase after the thinking phase
- Configurable model and dataset paths
- Outputs separate files for thinking process and final revisions

**Requirements:**

Install vLLM before running the script:

```bash
pip install vllm
```

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

## Citation

### Our Work

This work is part of our research on **empowering IETF collaboration through advanced NLP and LLM techniques**. Our paper presents a comprehensive framework that combines semantic search capabilities with large language model-enhanced writing assistance for RFC (Request for Comments) documents. The think-and/or-wait (tts) text generation approach demonstrated here is one of the key components of our LLM-enhanced writing system.

**Key contributions of our work:**
- Novel semantic search system for navigating IETF document collections
- LLM-powered text revision and enhancement for technical writing
- Integration of test-time scaling techniques for improved generation quality
- Practical tools for supporting collaborative standards development

For detailed results and evaluation, please see our paper:

```
@inproceedings{bian2025empowering,
  title={Empowering IETF Collaboration with NLP Search Innovations and LLM-Enhanced RFC Writing},
  author={Bian, Jie and Welzl, Michael},
  booktitle={Proceedings of the 2025 Applied Networking Research Workshop},
  pages={24--31},
  year={2025}
}
```

This implementation of **Test-Time Scaling (TTS)** is inspired by the **s1** framework, which introduces a simple yet effective approach to improving language model reasoning through test-time computation. The s1 method demonstrates that scaling inference-time computation can significantly enhance model performance on complex tasks without requiring additional training.

### Key Concepts from s1:

- **Test-time scaling**: Allocating more computational resources during inference rather than training
- **Multi-phase reasoning**: Breaking down the reasoning process into thinking, optional waiting, and answering phases
- **Iterative refinement**: Allowing models more time to deliberate on complex problems

The think-and-wait mechanism implemented here follows the s1 philosophy of giving models the opportunity to "think longer" before producing final outputs, which can lead to higher quality text revisions.

For more details, see the s1 repository: https://github.com/simplescaling/s1

### Reference:

```
@misc{muennighoff2025s1simpletesttimescaling,
      title={s1: Simple test-time scaling}, 
      author={Niklas Muennighoff and Zitong Yang and Weijia Shi and Xiang Lisa Li and Li Fei-Fei and Hannaneh Hajishirzi and Luke Zettlemoyer and Percy Liang and Emmanuel Candès and Tatsunori Hashimoto},
      year={2025},
      eprint={2501.19393},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.19393}, 
}
```