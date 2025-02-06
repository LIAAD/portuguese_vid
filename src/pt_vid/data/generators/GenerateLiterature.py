from datasets import Dataset
from pt_vid.data.generators.Generator import Generator

class GenerateLiterature(Generator):
    def generate(self)->Dataset:
        raise NotImplementedError('GenerateLiterature.generate is not implemented')