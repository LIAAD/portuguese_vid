from datasets import Dataset
from nltk.tokenize import word_tokenize

class DatasetStats:
    def __init__(self, dataset:Dataset):
        self.dataset = dataset

    def _token_counter(self, example, text_column="text"):
        n_tokens = len(word_tokenize(example["text"], language="portuguese"))
    
    def token_counter(self, text_column="text"):
        return self.dataset.map(lambda example: self._token_counter(example, text_column), batched=True)
    
    def 