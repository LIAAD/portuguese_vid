from transformers import BertTokenizerFast
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

BERT_MAX_LEN = 0

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

def tokenize(dataset):
    global BERT_MAX_LEN

    tokenizer = BertTokenizerFast.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", max_length=BERT_MAX_LEN)

    dataset = dataset.map(lambda example: tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=BERT_MAX_LEN))

    return dataset


def pre_process_dataset(domain, bert_max_len):
    global BERT_MAX_LEN

    BERT_MAX_LEN = bert_max_len
    
    if domain == 'dslcc':
        dataset = load_dataset("arubenruben/portuguese_dslcc")
    else:
        dataset = load_dataset("arubenruben/Portuguese_Language_Identification", domain)
    
    for split in ['train', 'test']:
        
        if split == 'train':
            dataset[split] = balance_data(dataset[split])
        
        dataset[split] = tokenize(dataset[split])

        #if split == 'train':
         #   dataset[split] = dataset[split].shuffle().select(range(min(len(dataset[split]), 30_000)))
        dataset[split] = dataset[split].shuffle().select(range(min(len(dataset[split]), 30_000)))
                     
        dataset[split].set_format(type='torch', columns=[
                                  'input_ids', 'attention_mask', 'label'])

    train_dataloader = create_dataloader(dataset['train'], shuffle=True)

    test_dataloader = create_dataloader(
        dataset['test'], shuffle=False)

    return train_dataloader, test_dataloader



def create_dataloader(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=8, shuffle=shuffle, num_workers=8, drop_last=True)


def load_test_set(domain):
    dataset = load_dataset(
        'arubenruben/Portuguese_Language_Identification', domain, split='test')

    dataset.set_format(type='torch', columns=['text', 'label'])

    dataset = tokenize(dataset)

    dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])

    test_dataloader = create_dataloader(dataset)

    return test_dataloader
