import logging

import pandas as pd
import datasets
from imblearn.under_sampling import RandomUnderSampler

from ptvid.constants import DOMAINS, N_PROC
from ptvid.src.delexicalizer import Delexicalizer


class Data:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def _balance_dataset(self, dataset) -> datasets.Dataset:
        df_dataset = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})
        logging.info(f"Class Balance before undersampling: {df_dataset['label'].value_counts()}")
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(df_dataset["text"].to_numpy().reshape(-1, 1), df_dataset["label"].to_numpy())
        df_dataset = pd.DataFrame({"text": X_res.reshape(-1), "label": y_res})
        logging.info(f"Class Balance after undersampling: {df_dataset['label'].value_counts()}")
        return datasets.Dataset.from_pandas(df_dataset)

    def _load_domain_all(self):
        return datasets.concatenate_datasets(
            [datasets.load_dataset(self.dataset_name, domain, split="train") for domain in DOMAINS]
        )

    def load_domain(
        self, domain: str, balance: bool, pos_prob: float, ner_prob: float, sample_size: int = None
    ) -> datasets.Dataset:
        delexicalizer = Delexicalizer(pos_prob, ner_prob)

        logging.info(f"Loading {domain} dataset")
        if domain == "all":
            dataset = self._load_domain_all()
        else:
            dataset = datasets.load_dataset(self.dataset_name, domain, split="train")

        if balance:
            logging.info("Balancing Training Dataset")
            dataset = self._balance_dataset(dataset)

        if sample_size is not None:
            logging.info("Sampling Training Dataset")
            dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))

        logging.info("Delexicalizing Training Dataset")
        dataset = dataset.map(lambda x: {"text": delexicalizer.delexicalize(x["text"])}, num_proc=N_PROC)
        return dataset

    def load_test_set(self) -> dict:
        return {domain: datasets.load_dataset(self.dataset_name, domain, split="valid") for domain in DOMAINS}
