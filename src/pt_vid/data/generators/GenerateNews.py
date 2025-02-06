from datasets import Dataset
from pt_vid.data.generators.Generator import Generator

class GenerateNews(Generator):
    def generate(self)->Dataset:
        raise NotImplementedError('GenerateSocialMedia.generate is not implemented')