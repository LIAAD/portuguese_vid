from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GeneratePolitics(Generator):
    def generate(self)->VIDDataset:
        europarl = load_dataset('arubenruben/europarl', split='train')
        
        brazilian_senate = load_dataset('arubenruben/brazilian_senate_speeches')

        return VIDDataset(
            dataset=concatenate_datasets([europarl, brazilian_senate['train'], brazilian_senate['test']]).shuffle(seed=42),
            config_name='politics'
        )

    # TODO: Migrate Code from others.py towards here