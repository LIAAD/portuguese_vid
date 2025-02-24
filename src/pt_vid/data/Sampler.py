import pandas as pd
from datasets import Dataset, DatasetDict
from pt_vid.data.Tokenizer import Tokenizer
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

tqdm.pandas()

class Sampler:
    def run(dataset_dict):

        dataset = dataset_dict['train']

        dataset = dataset.shuffle(seed=42)

        dataset = dataset.to_pandas()

        rus = RandomUnderSampler(random_state=42)

        dataset, _ = rus.fit_resample(dataset, dataset['label'])

        dataset['token_count'] = dataset['text'].progress_apply(
            lambda text : len(Tokenizer.tokenize(text)))

        q1 = dataset['token_count'].quantile(0.25)
        q2 = dataset['token_count'].quantile(0.5)
        q3 = dataset['token_count'].quantile(0.75)

        under_q1_sample = dataset[dataset['token_count'] < q1].sample(5)
        between_q1_q2_sample = dataset[(dataset['token_count'] >= q1) & (
            dataset['token_count'] < q2)].sample(20)

        between_q2_q3_sample = dataset[(dataset['token_count'] >= q2) & (
            dataset['token_count'] < q3)].sample(20)
        
        over_q3_sample = dataset[dataset['token_count'] > q3].sample(5)

        sample = pd.concat(
            [under_q1_sample, between_q1_q2_sample, between_q2_q3_sample, over_q3_sample]
        )

        return DatasetDict({
            'train': Dataset.from_pandas(sample, split='train'),
            'test': dataset_dict['test']
        })