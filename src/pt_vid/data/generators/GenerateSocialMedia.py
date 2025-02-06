from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GenerateSocialMedia(Generator):
    def generate(self)->VIDDataset:
        raw_dataset = load_dataset('arubenruben/hate_br_li')

        return VIDDataset(
            dataset=concatenate_datasets([raw_dataset['train'], raw_dataset['test']]),
            config_name='social_media'
        )