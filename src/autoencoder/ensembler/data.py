from datasets import Dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader


def tokenize(dataset):
    BERT_MAX_LEN = 512

    tokenizer = BertTokenizerFast.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", max_length=BERT_MAX_LEN)

    dataset = dataset.map(lambda example: tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=BERT_MAX_LEN))
    
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), 50_000)))
    
    return dataset


def create_dataloader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=128, shuffle=shuffle, num_workers=8, drop_last = True)
