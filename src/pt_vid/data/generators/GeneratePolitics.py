from datasets import Dataset
from pt_vid.data.generators.Generator import Generator

class GeneratePolitics(Generator):
    def generate(self)->Dataset:
        raise NotImplementedError('GenerateSocialMedia.generate is not implemented')