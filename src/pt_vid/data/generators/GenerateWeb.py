from datasets import load_dataset
from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator

class GenerateWeb(Generator):
    @staticmethod
    def generate()->VIDDataset:
        return VIDDataset(
            dataset=load_dataset('arubenruben/OSCAR-PT-BR-100K', split='train'),
            config_name='web'
        )