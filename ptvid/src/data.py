import logging

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from imblearn.under_sampling import RandomUnderSampler
from pt_variety_identifier.src.delexicalizer import Delexicalizer


class Data:
    def __init__(self, dataset_name) -> None:
        self._DOMAINS = ["journalistic", "literature", "legal", "politics", "web", "social_media"]
        self.dataset_name = dataset_name

    def balance_dataset(self, dataset):
        df_dataset = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})

        logging.info(f"Classe Balance Before Undersampling: {df_dataset['label'].value_counts()}")

        rus = RandomUnderSampler(random_state=42)

        X_res, y_res = rus.fit_resample(df_dataset["text"].to_numpy().reshape(-1, 1), df_dataset["label"].to_numpy())

        df_dataset = pd.DataFrame({"text": X_res.reshape(-1), "label": y_res})

        logging.info(f"Classe Balance After Undersampling: {df_dataset['label'].value_counts()}")

        return Dataset.from_pandas(df_dataset)

    def _load_domain_all(self):
        dataset_return = None

        for domain in self._DOMAINS:
            dataset = load_dataset(self.dataset_name, domain, split="train")

            if dataset_return is None:
                dataset_return = dataset
            else:
                dataset_return = concatenate_datasets([dataset_return, dataset])

        return dataset_return

    def load_domain(self, domain, balance, pos_prob, ner_prob, sample_size=None):
        delexicalizer = Delexicalizer(pos_prob, ner_prob)

        logging.info(f"Loading {domain} dataset")

        if domain == "all":
            dataset = self._load_domain_all()
        else:
            dataset = load_dataset(self.dataset_name, domain, split="train")

        if balance:
            logging.info("Balancing Training Dataset")
            dataset = self.balance_dataset(dataset)

        if sample_size != None:
            logging.info("Sampling Training Dataset")
            dataset = dataset.shuffle(seed=42).select(range(sample_size))

        logging.info("Delexicalizing Training Dataset")

        df_train = dataset.to_pandas()

        df_train["text"] = df_train["text"].progress_apply(delexicalizer.delexicalize)

        return Dataset.from_pandas(df_train)

    def load_test_set(self):
        dataset_return = {}

        for domain in self._DOMAINS:
            dataset_return[domain] = load_dataset(self.dataset_name, domain, split="test")

        return dataset_return
