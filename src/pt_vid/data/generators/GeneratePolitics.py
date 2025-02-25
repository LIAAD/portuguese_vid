from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GeneratePolitics(Generator):
    @staticmethod
    def _generate_europarl():
        europarl = load_dataset('arubenruben/europarl', split='train')
        
        # Add label column
        europarl['label'] = [0] * len(europarl)

        # Remove first 2 and last 2 sentences of each text.
        # The goal is to remove greetings and farewells.
        europarl['text'] = [text.split('. ')[2:-2] for text in europarl['text']]

        return europarl

    @staticmethod
    def _generate_brazilian_senate():
        brazilian_senate = load_dataset('arubenruben/brazilian_senate_speeches')

        # Remove first 5 and last 5 sentences of each text.
        # The goal is to remove greetings and farewells.
        brazilian_senate['text'] = [text.split('. ')[5:-5] for text in brazilian_senate['text']]

        return brazilian_senate
    
    @staticmethod
    def generate()->VIDDataset:
        europarl = GeneratePolitics._generate_europarl()

        brazilian_senate = GeneratePolitics._generate_brazilian_senate()

        return VIDDataset(
            dataset=concatenate_datasets([europarl, brazilian_senate['train'], brazilian_senate['test']]).shuffle(seed=42),
            config_name='politics'
        )