from pathlib import Path

import datasets
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from ptvid.constants import CACHE_DIR
from ptvid.src.data import Data as BaseData


class Data(BaseData):
    def __init__(
        self, dataset_name: str, split: str, tokenizer_name: str, batch_size: int, cache_dir: Path = CACHE_DIR
    ):
        super().__init__(dataset_name=dataset_name, split=split, cache_dir=cache_dir)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.batch_size = batch_size

    def _tokenize(self, example):
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

    def _adapt_dataset(self, dataset):
        dataset = dataset.map(self._tokenize, batched=True)

        # Set the tensor type and the columns which the dataset should return
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(dataset, batch_size=self.batch_size)

    def load_domain(
        self,
        domain: str,
        balance: bool,
        pos_prob: float = 0.,
        ner_prob: float = 0.,
        sample_size: int = None,
    ) -> datasets.Dataset:
        dataset = super().load_domain(
            domain=domain, balance=balance, pos_prob=pos_prob, ner_prob=ner_prob, sample_size=sample_size
        )
        return self._adapt_dataset(dataset)
