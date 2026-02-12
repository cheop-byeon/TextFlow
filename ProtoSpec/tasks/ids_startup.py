import json
import re
import os
import logging
from typing import Dict, List, Any, Tuple
from evaluate import load
from ProtoSpec.base import Task
import csv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ORDER = 4
DEFAULT_SMOOTH = True
PROMPT_TEMPLATE = """You are a professional IETF RFC writer. 
Below is some feedback discussing changes needed for a text. 
Please provide a revised version of the text based solely on the feedback.

Input:

Feedback:
{reviews}

Output:

Revised Text:\n
"""

def avg_bert_score(bertscore_results: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate average BERTScore metrics.
    
    Args:
        bertscore_results: Dictionary containing precision, recall, and f1 scores
        
    Returns:
        Dictionary with averaged metrics
    """
    return {
        'avg_pre': np.mean(bertscore_results['precision']),
        'avg_rec': np.mean(bertscore_results['recall']),
        'avg_f1': np.mean(bertscore_results['f1'])
    }


class IdsTextFlow(Task):
    """Task for evaluating text revision quality using multiple metrics (feedback-only prompt)."""
    
    DEFAULT_DATASET_PATH = "./ids/ids.i2c.test.generation.jsonl"
    DATASET_SPLIT = "train"  # The split name used by the dataset loader
    
    def __init__(self, max_order: int = DEFAULT_MAX_ORDER, smooth: bool = DEFAULT_SMOOTH, dataset_path: str = None):
        """Initialize IdsBase2 task.
        
        Args:
            max_order: Maximum n-gram order for BLEU score computation
            smooth: Whether to use smoothing for BLEU score
            dataset_path: Path to the dataset file (optional, defaults to DEFAULT_DATASET_PATH)
        """
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )
        self.max_order = max_order
        self.smooth = smooth
        self.DATASET_PATH = dataset_path if dataset_path is not None else self.DEFAULT_DATASET_PATH
        logger.info(f"Initialized IdsBase2 with dataset: {self.DATASET_PATH}")

    def get_dataset(self):
        """Returns dataset for the task.
        
        Note: Despite the 'train' split name, this actually loads the test data
        from DATASET_PATH. The split name is determined by how the dataset loader
        organizes the data, not the actual purpose of the data.
        
        Returns:
            Dataset containing test examples
        """
        return self.dataset[self.DATASET_SPLIT]

    def get_prompt(self, doc: Dict[str, str]) -> str:
        """Generate prompt for the given document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Formatted prompt string
        """
        return self.generate_prompt(doc["comments"])

    def get_reference(self, doc: Dict[str, str]) -> str:
        """Builds the reference solution for the doc (sample from the test dataset).
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Reference text
        """
        return doc["new_text"]

    def get_old_text(self, doc: Dict[str, str]) -> str:
        """Extract old text from document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Original text
        """
        return doc["old_text"]

    def get_comments(self, doc: Dict[str, str]) -> str:
        """Extract comments from document.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Comments text
        """
        return doc["comments"]

    def postprocess_generation(self, generation: str, idx: int) -> str:
        """Extract and clean the revised text from generation.
        
        Args:
            generation: Generated text from model
            idx: Index of the generation
            
        Returns:
            Cleaned output text
        """
        output = generation.split("Revised Text:\n", 1)[-1].strip()
        return output

    @staticmethod
    def generate_prompt(reviews: str) -> str:
        """Generate prompt from feedback/reviews.
        
        Args:
            reviews: Feedback text discussing needed changes
            
        Returns:
            Formatted prompt
        """
        return PROMPT_TEMPLATE.format(reviews=reviews)

    def _filter_empty_generations(
        self, 
        generations: List[str], 
        references: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter out empty generations and their corresponding references.
        
        Args:
            generations: List of generated texts
            references: List of reference texts
            
        Returns:
            Tuple of filtered (generations, references)
        """
        filtered_pairs = [
            (gen, ref) 
            for gen, ref in zip(generations, references) 
            if gen.strip()
        ]
        
        if not filtered_pairs:
            logger.warning("All generations are empty!")
            return [], []
        
        gens, refs = zip(*filtered_pairs)
        logger.info(f"Filtered {len(generations)} generations to {len(gens)} non-empty ones")
        return list(gens), list(refs)

    def process_results(
        self, 
        generations: List[List[str]], 
        references: List[str]
    ) -> Dict[str, Any]:
        """Process results and compute multiple evaluation metrics.
        
        Args:
            generations: List of generation lists (each containing one generation)
            references: List of reference texts
            
        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Loading evaluation metrics...")
        
        # Load all metrics
        bleu = load("bleu")
        sacrebleu = load("sacrebleu")
        google_bleu = load("google_bleu")
        roberta_bertscore = load("bertscore")
        deberta_bertscore = load("bertscore", config_name="microsoft/deberta-xlarge-mnli")
        meteor = load('meteor')
        exact_match = load("exact_match")
        wer = load("wer")
        mauve = load('mauve')

        # Extract first generation from each list
        gens = [gen[0] for gen in generations]
        logger.info(f"Processing {len(gens)} generations")

        # Note: No filtering in this version - processes all generations
        logger.info("Computing metrics...")
        
        # Compute all metrics
        bleu_results = bleu.compute(
            references=references, 
            predictions=gens, 
            max_order=self.max_order, 
            smooth=self.smooth
        )
        sacre_bleu_results = sacrebleu.compute(
            references=references, 
            predictions=gens, 
            lowercase=True
        )
        google_bleu_results = google_bleu.compute(
            references=references, 
            predictions=gens
        )
        meteor_results = meteor.compute(
            references=references, 
            predictions=gens
        )
        exact_match_results = exact_match.compute(
            references=references, 
            predictions=gens, 
            ignore_case=True, 
            ignore_punctuation=True
        )
        wer_results = wer.compute(
            references=references, 
            predictions=gens
        )
        mauve_results = mauve.compute(
            references=references, 
            predictions=gens
        )
        
        # Compute BERTScore metrics
        logger.info("Computing BERTScore metrics...")
        roberta_bertscore_results = roberta_bertscore.compute(
            references=references, 
            predictions=gens, 
            lang="en"
        )
        avg_roberta_bertscore = avg_bert_score(roberta_bertscore_results)

        deberta_bertscore_results = deberta_bertscore.compute(
            references=references, 
            predictions=gens, 
            lang="en"
        )
        avg_deberta_bertscore = avg_bert_score(deberta_bertscore_results)

        logger.info("Evaluation complete")
        
        return {
            'bleu': bleu_results,
            'sacre_bleu': sacre_bleu_results,
            'google_bleu': google_bleu_results,
            'roberta_bertscore': roberta_bertscore_results,
            'roberta_bertscore_avg': avg_roberta_bertscore,
            'deberta_bertscore': deberta_bertscore_results,
            'deberta_bertscore_avg': avg_deberta_bertscore,
            'meteor': meteor_results,
            'EM': exact_match_results,
            'wer': wer_results,
            'mauve': mauve_results.mauve,
        }


