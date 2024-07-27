from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


class Spliter:
    def __init__(self, load_dataset_name, result_dataset_name) -> None:
        self.load_dataset_name = load_dataset_name
        self.result_dataset_name = result_dataset_name
        self.DEFAULT_SPLITS = {
            'journalistic': 10_000,
            'legal': 1_000,
            'literature': 5_000,
            'politics': 1_000,
            'social_media': 1_000,
            'web': 10_000,
        }

    def split(self, domain, num_test_samples=None):

        if num_test_samples is None:
            print(f"Using default number of test samples for {domain}")
            num_test_samples = self.DEFAULT_SPLITS[domain]

        dataset = load_dataset(self.load_dataset_name, domain, split='train')

        dataset = dataset.shuffle()

        dataset = dataset.to_pandas()

        # Extract num_test_samples/2 samples from each class. While doing so, remove them from the original dataset
        european_sample = dataset[dataset['label']
                                  == 0].sample(num_test_samples//2)
        brazilian_sample = dataset[dataset['label']
                                   == 1].sample(num_test_samples//2)

        dataset = dataset.drop(european_sample.index)
        dataset = dataset.drop(brazilian_sample.index)

        # Create the test dataset
        test_dataset = pd.concat(
            [european_sample, brazilian_sample], ignore_index=True)

        dataset = dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        # datasets columns must be text and label
        dataset = dataset[['text', 'label']]
        test_dataset = test_dataset[['text', 'label']]

        DatasetDict({
            'train': Dataset.from_pandas(dataset),
            'test': Dataset.from_pandas(test_dataset)
        }).push_to_hub(self.result_dataset_name, config_name=domain)


if __name__ == '__main__':
    for domain in ['politics', 'web', 'journalistic', 'legal', 'literature', 'social_media']:
        spliter = Spliter(
            load_dataset_name="arubenruben/portuguese-language-identification-cleaned",
            result_dataset_name="arubenruben/portuguese-language-identification",
        )
        spliter.split(domain)
