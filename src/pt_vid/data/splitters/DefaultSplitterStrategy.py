import os
import json
import pandas as pd
from datasets import DatasetDict, Dataset
from pt_vid.data.splitters.Strategy import Strategy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class DefaultSplitterStrategy(Strategy):
    DEFAULT_SPLITS: dict = json.load(open(os.path.join(CURRENT_DIR, 'default_test_splits.json')))
    
    @staticmethod
    def run(dataset, domain)->DatasetDict:
        dataset = dataset.shuffle(seed=42)

        dataset = dataset.to_pandas()
        
        num_test_samples = DefaultSplitterStrategy.DEFAULT_SPLITS[domain]

        european_sample = dataset[dataset['label'] == 0].sample(num_test_samples//2)
        brazilian_sample = dataset[dataset['label'] == 1].sample(num_test_samples//2)

        dataset = dataset.drop(european_sample.index)
        dataset = dataset.drop(brazilian_sample.index)

        test_dataset = pd.concat([european_sample, brazilian_sample], ignore_index=True)

        dataset = dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        dataset = dataset[['text', 'label']]
        test_dataset = test_dataset[['text', 'label']]

        return DatasetDict({
            'train': Dataset.from_pandas(dataset, split='train'),
            'test': Dataset.from_pandas(test_dataset, split='test')
        })