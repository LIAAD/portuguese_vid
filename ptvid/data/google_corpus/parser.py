import os
import pandas as pd
from datasets import Dataset, DatasetDict

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Read tsv file
pt_pt = pd.read_csv(os.path.join(
    CURRENT_PATH, 'pt_entity_test_en_pt-PT.tsv'), sep='\t', header=None)
pt_br = pd.read_csv(os.path.join(
    CURRENT_PATH, 'pt_entity_test_en_pt-BR.tsv'), sep='\t', header=None)

# Create new column label
pt_pt['label'] = [0] * len(pt_pt)
pt_br['label'] = [1] * len(pt_br)

# Force column names to be: English, Portuguese
pt_pt.columns = ['en', 'text', 'label']
pt_br.columns = ['en', 'text', 'label']

# Drop English column
pt_pt = pt_pt.drop('en', axis=1)
pt_br = pt_br.drop('en', axis=1)

pt_df = pd.concat([pt_pt, pt_br], ignore_index=True)

DatasetDict({
    'test': Dataset.from_pandas(pt_df).shuffle(seed=42)
}).push_to_hub('arubenruben/google_research_FRMT_Portuguese_Variety_Identifier')
