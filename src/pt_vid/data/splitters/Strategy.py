from datasets import DatasetDict
from abc import ABC, abstractmethod

class Strategy(ABC):
    @staticmethod
    @abstractmethod
    def run(dataset, domain)->DatasetDict:
        raise NotImplementedError