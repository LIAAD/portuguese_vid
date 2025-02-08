from pt_vid.entity.VIDDataset import VIDDataset
from pt_vid.data.generators.Generator import Generator
from datasets import load_dataset, concatenate_datasets

class GenerateLaw(Generator):
    @staticmethod
    def _generate_lener_br():
        dataset = load_dataset('lener_br')

        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])

        dataset['label'] = [1] * len(dataset)

        dataset['text'] = dataset['tokens'].apply(lambda x: ' '.join(x))

        # Remove any other columns than text and label
        dataset = dataset.select_columns(['text', 'label'])

        return dataset
    
    @staticmethod
    def _generate_dgsi():
        dataset = load_dataset("stjiris/portuguese-legal-sentences-v0")
        
        dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])

        # Rename sentence column to text
        dataset = dataset.rename_column('sentence', 'text')

        # Add label column
        dataset['label'] = [0] * len(dataset)

        dataset = dataset.select_columns(['text', 'label'])

        return dataset


    @staticmethod
    def generate()->VIDDataset:
        lener_br = GenerateLaw._generate_lener_br()

        dgsi = GenerateLaw._generate_dgsi()

        return VIDDataset(
            dataset=concatenate_datasets([lener_br, dgsi]).shuffle(seed=42),
            config_name='law'
        )
    