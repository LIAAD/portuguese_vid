import os
import pandas as pd
from datasets import Dataset, DatasetDict

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read tsv file
pt_train = pd.read_csv(os.path.join(
    CURRENT_PATH, 'PT_train.tsv'), sep='\t', header=None)

pt_dev = pd.read_csv(os.path.join(
    CURRENT_PATH, 'PT_dev.tsv'), sep='\t', header=None)

# force column names to be: id, text, label
pt_train.columns = ['id', 'text', 'label']
pt_dev.columns = ['id', 'text', 'label']

# Index column id
pt_train = pt_train.set_index('id')
pt_dev = pt_dev.set_index('id')

# PT-BR: 1, PT-PT: 0, PT: 2
pt_train['label'] = pt_train['label'].map({'PT-BR': 1, 'PT-PT': 0, 'PT': 2})
pt_dev['label'] = pt_dev['label'].map({'PT-BR': 1, 'PT-PT': 0, 'PT': 2})

#drop the index column
pt_train = pt_train.reset_index(drop=True)
pt_dev = pt_dev.reset_index(drop=True)

dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pt_train).shuffle(seed=42),
    'validation': Dataset.from_pandas(pt_dev).shuffle(seed=42)
}).push_to_hub('arubenruben/portuguese_dsl_tl')
