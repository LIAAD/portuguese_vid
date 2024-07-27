from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler


class Sampler:
    def __init__(self, hf_dataset_name, domains) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.domains = domains
        self.rus = RandomUnderSampler(random_state=42)
        tqdm.pandas()

    def sample(self):

        df_final = pd.DataFrame(columns=['sentence', 'label', 'domain'])

        for domain in self.domains:
            dataset = load_dataset(self.hf_dataset_name, domain, split='train')

            dataset = dataset.shuffle()

            dataset = dataset.to_pandas()

            # Undersample the dataset
            dataset, _ = self.rus.fit_resample(dataset, dataset['label'])

            # Create Token Count column
            print(f"Calculating token count for {domain}...")
            dataset['token_count'] = dataset['text'].progress_apply(
                lambda text: len(text.split()))

            # Calculate the Quantiles for the Token Count column

            q1 = dataset['token_count'].quantile(0.25)
            q2 = dataset['token_count'].quantile(0.5)
            q3 = dataset['token_count'].quantile(0.75)

            # Sample 5 sentences with token counter lower than q1
            under_q1_sample = dataset[dataset['token_count'] < q1].sample(5)

            # Sample 20 sentences with token counter between q1 and q2
            between_q1_q2_sample = dataset[(dataset['token_count'] >= q1) & (
                dataset['token_count'] < q2)].sample(20)

            # Sample 20 sentences with token counter between q2 and q3
            between_q2_q3_sample = dataset[(dataset['token_count'] >= q2) & (
                dataset['token_count'] < q3)].sample(20)

            # Sample 5 sentences with token counter higher than q3
            over_q3_sample = dataset[dataset['token_count'] > q3].sample(5)

            # Concatenate all samples
            sample = pd.concat(
                [under_q1_sample, between_q1_q2_sample, between_q2_q3_sample, over_q3_sample])

            # Add the domain column
            sample['domain'] = [domain] * sample.shape[0]

            df_final = pd.concat([df_final, sample], ignore_index=True)

        return df_final

if __name__ == '__main__':
    sampler = Sampler(
        hf_dataset_name="arubenruben/portuguese-language-identification-splitted",
        domains=['politics', 'web', 'journalistic',
                 'legal', 'literature', 'social_media']
    )

    df_sample = sampler.sample()

    df_sample.to_csv('sample.csv', index=False)