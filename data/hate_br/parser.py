from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, ClassLabel, DatasetDict
import dotenv
import os
import demoji
import re

dotenv.load_dotenv(dotenv.find_dotenv())
dataset = load_dataset('ruanchaves/hatebr')


dataset['train'] = concatenate_datasets(
    [dataset['train'], dataset['validation']])

dataset = dataset.select_columns('instagram_comments')

dataset = dataset.rename_column('instagram_comments', 'text')

dataset_dict = DatasetDict({'train': None, 'test': None})

for split in ['train', 'test']:

    df = dataset[split].to_pandas()

    df['label'] = 'pt-BR'

    for index, row in df.iterrows():
        text = row['text'].strip()
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = demoji.replace(text, '')

        df.at[index, 'text'] = text

    dataset_dict[split] = Dataset.from_pandas(df, split='train', features=Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
    }))


dataset_dict.push_to_hub('hate_br_li', token=os.getenv("HF_TOKEN"))
