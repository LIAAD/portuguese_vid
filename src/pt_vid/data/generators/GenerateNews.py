from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GenerateNews(Generator):
    def generate(self)->VIDDataset:
        dataset_dict = load_dataset('arubenruben/cetem')

        return VIDDataset(
            raw_dataset=concatenate_datasets([dataset_dict['train'], dataset_dict['test']])
        )