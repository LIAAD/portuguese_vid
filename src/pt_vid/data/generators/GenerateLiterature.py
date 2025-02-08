from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GenerateLiterature(Generator):
    @staticmethod
    def generate()->VIDDataset:
        raw_dataset = load_dataset('arubenruben/brazilian_literature')

        return VIDDataset(
            dataset=concatenate_datasets([raw_dataset['train'], raw_dataset['test']]),
            config_name='literature'
        )