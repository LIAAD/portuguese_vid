from datasets import Dataset
from pt_vid.data.generators.Generator import Generator

class GenerateWeb(Generator):
    def generate(self)->Dataset:
        raise NotImplementedError('GenerateWeb.generate is not implemented')