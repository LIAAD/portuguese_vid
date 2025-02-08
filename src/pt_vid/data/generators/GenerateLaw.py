from datasets import Dataset, load_dataset
from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator

class GenerateLaw(Generator):
    def generate(self)->VIDDataset:
        raise NotImplementedError('GenerateLaw.generate is not implemented')

    # TODO: Migrate Code from others.py towards here