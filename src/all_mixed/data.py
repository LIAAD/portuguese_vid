from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset
from imblearn.under_sampling import RandomUnderSampler
from datasets import Dataset, concatenate_datasets
import pandas as pd

BERT_MAX_LEN=512

def tokenize(dataset):
    global BERT_MAX_LEN

    tokenizer = BertTokenizerFast.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", max_length=BERT_MAX_LEN)

    dataset = dataset.map(lambda example: tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=BERT_MAX_LEN))

    return dataset


def balance_data(dataset):

    df = dataset.to_pandas()

    print(f"Before balancing: {df['label'].value_counts()}")

    rus = RandomUnderSampler(random_state=42, replacement=True)

    X_resampled, y_resampled = rus.fit_resample(
        df['text'].to_numpy().reshape(-1, 1), df['label'].to_numpy())

    df = pd.DataFrame(
        {'text': X_resampled.flatten(), 'label': y_resampled})

    print(f"After balancing: {df['label'].value_counts()}")

    return Dataset.from_pandas(df)


def create_dataloader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=8, drop_last=True)


def pre_process_dataset(bert_max_len):
    global BERT_MAX_LEN

    BERT_MAX_LEN = bert_max_len

    final_dataset = None

    for domain in ['literature', 'web', 'politics', 'social_media', 'law', 'news']:

        dataset = load_dataset(
            "arubenruben/Portuguese_Language_Identification", domain, split="train")

        dataset = balance_data(dataset)

        dataset = tokenize(dataset)

        dataset = dataset.shuffle().select(
            range(min(len(dataset), 30_000)))

        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        final_dataset = concatenate_datasets(
            [final_dataset, dataset]) if final_dataset is not None else dataset


    train_dataloader = create_dataloader(final_dataset, shuffle=True)

    return train_dataloader


def load_test_set():

    final_dataset = None

    for domain in ['politics', 'news', 'law', 'social_media', 'literature', 'web']:

        dataset = load_dataset(
            'arubenruben/Portuguese_Language_Identification', domain, split='test')

        dataset.set_format(type='torch', columns=['text', 'label'])

        dataset = dataset.shuffle().select(range(min(len(dataset), 50_000)))

        dataset = tokenize(dataset)

        dataset.set_format(type='torch', columns=[
                           'input_ids', 'attention_mask', 'label'])

        final_dataset = concatenate_datasets(
            [final_dataset, dataset]) if final_dataset is not None else dataset

    test_dataloader = create_dataloader(dataset)

    return test_dataloader


def load_test_set_with_domain(domain):
    
    dataset = load_dataset(
        'arubenruben/Portuguese_Language_Identification', domain, split='test')
    
    dataset.set_format(type='torch', columns=['text', 'label'])
    
    dataset = dataset.shuffle().select(range(min(len(dataset), 50_000)))
    
    dataset = tokenize(dataset)
    

    dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])
    
    test_dataloader = create_dataloader(dataset)

    return test_dataloader