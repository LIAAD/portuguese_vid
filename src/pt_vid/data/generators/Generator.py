from abc import ABC, abstractmethod
from pt_vid.entity.VIDDataset import VIDDataset

class Generator(ABC):
    @abstractmethod
    def generate(self)->VIDDataset:
        raise NotImplementedError('Generator.generate is not implemented')