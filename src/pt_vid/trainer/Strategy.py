from datasets import Dataset
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, training_dataset:Dataset, training_datasets_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_dataset = training_dataset
        self.training_datasets_names = training_datasets_names

    @abstractmethod
    def train():
        raise NotImplementedError