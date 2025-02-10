from datasets import Dataset
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, training_dataset:Dataset, validation_dataset:Dataset=None, eval_dataset:Dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.eval_dataset = eval_dataset
    
    @abstractmethod
    def train():
        raise NotImplementedError