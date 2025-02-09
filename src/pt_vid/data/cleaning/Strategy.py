from datasets import Dataset
from abc import ABC, abstractmethod

class Strategy(ABC):
    @staticmethod
    @abstractmethod
    def _run(row)->str:
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def run(dataset:Dataset)->Dataset:
        raise NotImplementedError("Subclasses must implement this method")