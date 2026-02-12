from abc import ABC, abstractmethod
from warnings import warn

from datasets import load_dataset


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """Initialize a task.
        
        Args:
            stop_words: List of stop words for generation stopping criteria (optional)
            requires_execution: Whether task requires code execution during evaluation (default: True)
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset('json', data_files=self.DATASET_PATH)
        except Exception as e:
            warn(f"Failed to load dataset from {self.DATASET_PATH}: {e}. Falling back to locally downloaded dataset.")

    @abstractmethod
    def get_dataset(self):
        """Return dataset or iterable of objects compatible with get_prompt()."""
        return []


    @abstractmethod
    def get_prompt(self, doc):
        """Build prompt for language model generation.
        
        Args:
            doc: Sample from test dataset (dict)
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Get reference solution for a sample.
        
        Args:
            doc: Sample from test dataset (dict)
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Postprocess language model generation.
        
        Args:
            generation: Generated text from LM (str)
            idx: Index of sample in dataset (int)
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Evaluate generations against references.
        
        Args:
            generations: List of generation lists (list[list[str]])
            references: List of reference solutions (list[str])
            
        Returns:
            Metrics dict with format {"metric_name": float_value}
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """Truncate string at first occurrence of any stop token.
        
        Args:
            decoded_string: Text to truncate (should not include prompt)
            stop_tokens: List of stop words to check
            
        Returns:
            Truncated string up to first stop token
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
