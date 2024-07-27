import torch
from transformers import BertTokenizerFast
from pt_variety_identifier.src.data import Data as BaseData
from torch.utils.data import DataLoader


class Data(BaseData):
    def __init__(self, dataset_name, tokenizer_name, batch_size):
        super().__init__(dataset_name=dataset_name)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.batch_size = batch_size

    def _tokenize(self, example):
        return self.tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

    def _adapt_dataset(self, dataset):
        dataset = dataset.map(self._tokenize, batched=True)

        # Set the tensor type and the columns which the dataset should return
        dataset.set_format(type='torch', columns=[
                           'input_ids', 'attention_mask', 'label'])

        return DataLoader(dataset, batch_size=self.batch_size)

    def load_domain(self, domain, balance, pos_prob, ner_prob, sample_size=None):
        dataset = super().load_domain(domain=domain, balance=balance,
                                      pos_prob=pos_prob, ner_prob=ner_prob, sample_size=sample_size)

        return self._adapt_dataset(dataset)

    def load_test_set(self):
        dataset_dict = super().load_test_set()

        for domain in dataset_dict.keys():
            dataset_dict[domain] = self._adapt_dataset(dataset_dict[domain])

        return dataset_dict
