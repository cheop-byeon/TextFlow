from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers import GenerationConfig, AutoConfig
from datasets import load_dataset
import json
import re
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text with optional thinking and waiting phases")
    parser.add_argument(
        "--use_wait",
        action="store_true",
        default=False,
        help="Whether to use the 'Wait' phase after thinking (default: False)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        help="Path to the model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../dataset/ids.i2c.test.generation.jsonl",
        help="Path to the dataset"
    )
    args = parser.parse_args()

    model_path = args.model_path
    model = LLM(model=model_path, quantization="gptq", tensor_parallel_size=1)

    tok = AutoTokenizer.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    MAX_TOKENS = config.max_position_embeddings
    MAX_TOKENS_THINKING = MAX_TOKENS - 768
    MAX_TOKENS_THINKING = 4096
    NUM_IGNORE = 1
    res1, res2 = [], []

    def contains_only_special_chars(text):
        """Check if text contains only whitespace and special characters.
        
        Args:
            text: Input text string
            
        Returns:
            Empty string if only special chars, otherwise original text
        """
        return "" if not text or not text.strip() or re.fullmatch(r'\W+', text.strip()) else text

    def extract_code_block(text, language=None):
        """Extract content from code blocks (``` or ---).
        
        Args:
            text: Input text containing code blocks
            language: Optional language identifier (e.g., 'markdown')
            
        Returns:
            Extracted content or original text if no blocks found
        """
        # Try language-specific markdown block first
        if language:
            pattern = rf'```{language}\s*(.*?)```'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try generic triple backticks
        match = re.search(r'```(?:\w+)?\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try triple dashes
        match = re.search(r'---\s*(.*?)---', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text

    def clean_generated_text(text, remove_markers=True, remove_newlines=False):
        """Clean generated text by removing markers and special characters.
        
        Args:
            text: Input text to clean
            remove_markers: Remove code block markers (```, ---)
            remove_newlines: Remove all newline characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Extract from code blocks if present
        cleaned = extract_code_block(text)
        
        if remove_markers:
            cleaned = cleaned.replace("```", "").replace("---", "")
        
        if remove_newlines:
            cleaned = cleaned.replace('\n', ' ').strip()
        
        # Remove if only special characters remain
        return contains_only_special_chars(cleaned)

    gen_kwargs = {
        #"do_sample": True,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "top_k": generation_config.top_k,
        "repetition_penalty": generation_config.repetition_penalty,
        #"max_new_tokens": max_new_tokens,
    }

    print("gen_kwargs", gen_kwargs)
    print(f"use_wait: {args.use_wait}")
    
    DATASET_PATH = args.dataset_path
    dataset = load_dataset('json', data_files=DATASET_PATH)
    dataset = dataset['train']#.select(range(10))

    def generate_prompt(old_text, reviews):

        prompt = f"""
        You are an experienced IETF RFC author. 
        Thoroughly review the original text snippet within an internet draft and feedback from the corresponding working group. 
        Identify and highlight parts needing revision. Assess the quality of the feedback. If the feedback is unclear, low-quality, or irrelevant, use your expertise to make independent judgments. 
        For complex feedback, take your time to carefully consider what it refers to and thoughtfully incorporate it into the original text. Finally provide well-considered revisions only.
        Original Text:
        {old_text}
        Feedback:
        {reviews}
        """

        return prompt

    for i, p in enumerate(dataset):
        prompt = """<|im_start|>user\n""" + generate_prompt(p['old_text'], p['comments']) + """<|im_end|>\n<|im_start|>assistant\n"""

        stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]

        sampling_params = SamplingParams(
            seed=42,
            max_tokens=MAX_TOKENS_THINKING,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature = generation_config.temperature,
            top_p = generation_config.top_p,
            top_k = generation_config.top_k,
            repetition_penalty = generation_config.repetition_penalty,
        )

        prompt += "<|im_start|>think"
        o = model.generate(
            prompt,
            sampling_params=sampling_params
        )

        max_tokens_thinking_tmp = MAX_TOKENS_THINKING # euqal or less than MAX_TOKENS
        max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)

        # Optional wait phase
        if args.use_wait and max_tokens_thinking_tmp > 0:
            ignore_str = "\nWait"
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                seed=42,
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature = generation_config.temperature,
                top_p = generation_config.top_p,
                top_k = generation_config.top_k,
                repetition_penalty = generation_config.repetition_penalty,
            )

            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )

            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
        else:
            # Skip wait phase, use thinking output directly
            prompt += o[0].outputs[0].text

        ### Final answer ###
        prompt += "\nFinal Revised Text:\n"
        stop_token_ids = tok("<|im_end|>")["input_ids"]


        sampling_params = SamplingParams(
            seed=42,
            max_tokens=256,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature = generation_config.temperature,
            top_p = generation_config.top_p,
            top_k = generation_config.top_k,
            repetition_penalty = generation_config.repetition_penalty,
        )
        o = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        print(prompt)
        print("*" * 10)
        print(o[0].outputs[0].text)
        print("^" * 10)

        start = prompt.find("<|im_start|>think") + len("<|im_start|>think")
        end = prompt.find("\nFinal Revised Text:\n")
        o1 = prompt[start:end].strip()

        o2 = o[0].outputs[0].text.strip()
        # Optional: Apply text cleaning functions
        # o2 = extract_code_block(o2, language='markdown')
        # o2 = clean_generated_text(o2, remove_markers=True, remove_newlines=False)
        res1.append(o1)
        res2.append(o2)

    print(len(res1), len(res2))

    formatted_strings = [[s] for s in res1]
    output_suffix = "_with_wait" if args.use_wait else "_no_wait"
    with open(f"think_qwen_32b{output_suffix}.json", "a") as f:
        f.write(json.dumps(formatted_strings))

    formatted_strs = [[s] for s in res2]
    with open(f"revised_qwen_32b{output_suffix}.json", "a") as f:
        f.write(json.dumps(formatted_strs))

if __name__ == "__main__":
    main()
