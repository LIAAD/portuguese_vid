import torch
from transformers import BertTokenizerFast
from datasets import load_dataset


def tokenize(batch):
    tokenizer = BertTokenizerFast.from_pretrained(
        'neuralmind/bert-base-portuguese-cased')

    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)


def process_data(dataset):
    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    return dataset

def create_dataloader(dataset, batch_size: int = 32):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def load_train_data(domain):
    
    if domain == 'dslcc':
        dataset = load_dataset('arubenruben/portuguese_dslcc')
    else:
        dataset = load_dataset('arubenruben/Portuguese_Language_Identification', domain)
    
    brazilian_data = {
        'train': dataset['train'].filter(lambda example: example['label'] == 1)
    }

    european_data = {
        'train': dataset['train'].filter(lambda example: example['label'] == 0)
    }

    brazilian_train = process_data(brazilian_data['train'])
    european_train = process_data(european_data['train'])

    brazilian_train = brazilian_train.shuffle(seed=42).select(range(min(len(brazilian_train), 50_000)))
    
    european_train = european_train.shuffle(seed=42).select(range(min(len(european_train), 50_000)))
 
    return create_dataloader(brazilian_train, batch_size=32), create_dataloader(european_train, batch_size=32)


def load_test_data(domain):
    
    if domain == 'dslcc':
        dataset = load_dataset('arubenruben/portuguese_dslcc')
    else:
        dataset = load_dataset('arubenruben/Portuguese_Language_Identification', domain)
    
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(min(len(dataset['test']), 50_000)))

    test_dataset = process_data(dataset['test'])
    
    return create_dataloader(test_dataset, batch_size=32)