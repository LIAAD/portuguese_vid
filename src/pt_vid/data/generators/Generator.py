from datasets import Dataset
from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def generate(self)->Dataset:
        raise NotImplementedError('Generator.generate is not implemented')