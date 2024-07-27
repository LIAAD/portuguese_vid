from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, Features, Value, ClassLabel
import pandas as pd
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())


def load_data():
    lener_br = load_dataset('lener_br')

    lener_br['train'] = concatenate_datasets(
        [lener_br['train'], lener_br['validation']])

    lener_br = lener_br.select_columns(
        ['tokens']).rename_column('tokens', 'text')

    df_train = pd.DataFrame(lener_br['train'], columns=['text'])

    df_train['label'] = 'pt-BR'

    df_test = pd.DataFrame(lener_br['test'], columns=['text'])

    df_test['label'] = 'pt-BR'

    return DatasetDict({'train': Dataset.from_pandas(df_train, split='train'), 'test': Dataset.from_pandas(df_test, split='test')})


def rebuild_sentences(dataset):

    features = Features({
        'text': Value('string'),
        "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
    })

    final_dataset = DatasetDict({
        'train': [],
        'test': []
    })

    for split in ['train', 'test']:
        new_dataset = {
            'text': [],
            'label': []
        }
        for row in dataset[split]:
            new_dataset['text'].append(' '.join(row['text']))
            new_dataset['label'].append(row['label'])

        final_dataset[split] = Dataset.from_dict(
            new_dataset, split=split, features=features)

    return final_dataset


if __name__ == "__main__":
    dataset = load_data()

    dataset = rebuild_sentences(dataset)

    dataset.push_to_hub(f'lener_br_language_id', token=os.getenv("HF_TOKEN"))
