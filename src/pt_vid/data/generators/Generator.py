from abc import ABC, abstractmethod
from pt_vid.entity.VIDDataset import VIDDataset

class Generator(ABC):
    @staticmethod
    @abstractmethod
    def generate()->VIDDataset:
        raise NotImplementedError('Generator.generate is not implemented')