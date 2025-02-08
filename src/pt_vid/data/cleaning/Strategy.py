from datasets import Dataset
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    @staticmethod
    def _run(text:str)->str:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    @staticmethod
    def run(dataset:Dataset):
        raise NotImplementedError("Subclasses must implement this method")