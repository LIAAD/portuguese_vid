import logging
from pathlib import Path

import datasets
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from ptvid.constants import CACHE_DIR, DOMAINS, N_PROC
from ptvid.src.delexicalizer import Delexicalizer


class Data:
    def __init__(self, dataset_name: str, split: str, cache_dir: Path = CACHE_DIR) -> None:
        self.dataset_name = dataset_name
        self._split = split
        self._cache_dir = cache_dir

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
            [datasets.load_dataset(self.dataset_name, domain, split=self._split) for domain in DOMAINS]
        )

    def _load_from_cache(self, cache_key: str):
        cache_path = self._cache_dir / cache_key
        if cache_path.exists():
            logging.info(f"Loading {cache_key} dataset from cache")
            dataset = datasets.load_from_disk(cache_path)
            return dataset
        return None

    def _save_to_cache(self, cache_key: str, dataset: datasets.Dataset):
        cache_path = self._cache_dir / cache_key
        cache_path.mkdir(exist_ok=True, parents=True)
        logging.info(f"Saving {cache_key} dataset to cache")
        dataset.save_to_disk(cache_path)

    def load_domain(
        self,
        domain: str,
        balance: bool,
        pos_prob: float = 0.,
        ner_prob: float = 0.,
        sample_size: int = None,
    ) -> datasets.Dataset:
        cache_key = f"{domain}_{self._split}_{balance}_{pos_prob}_{ner_prob}_{sample_size}"
        cached_dataset = self._load_from_cache(cache_key)
        if cached_dataset is not None:
            return cached_dataset

        logging.info(f"Loading {domain} dataset")
        if domain == "all":
            dataset = self._load_domain_all()
        else:
            dataset = datasets.load_dataset(self.dataset_name, domain, split=self._split)

        if balance:
            logging.info("Balancing Training Dataset")
            dataset = self._balance_dataset(dataset)

        if sample_size is not None:
            logging.info("Sampling Training Dataset")
            dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))

        logging.info("Delexicalizing Training Dataset")
        if pos_prob > 0 or ner_prob > 0:
            delexicalizer = Delexicalizer(pos_prob, ner_prob)
            dataset = dataset.map(lambda x: {"text": delexicalizer.delexicalize(x["text"])}, num_proc=N_PROC)

        self._save_to_cache(cache_key, dataset)
        return dataset
