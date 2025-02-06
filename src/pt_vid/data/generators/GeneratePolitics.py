from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GeneratePolitics(Generator):
    def generate(self)->VIDDataset:
        europarl = load_dataset('arubenruben/europarl', split='train')
        brazilian_senate = load_dataset('arubenruben/brazilian_senate')

        return VIDDataset(
            raw_dataset=concatenate_datasets([europarl, brazilian_senate['train'], brazilian_senate['test']]).shuffle(seed=42)
        )