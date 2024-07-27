from pathlib import Path
import os
import dotenv
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel, DatasetInfo
import pandas as pd

dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent


def readfiles():
    corpora = []
    data_path = os.path.join(CURRENT_PATH, 'data')

    for root, directories, files in os.walk(data_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    corpora.append(f.read().strip(). replace('\n', ' '))

    return corpora


def build_dataset(list_files):
    df = pd.DataFrame(list_files, columns=['text'])

    df['label'] = 'pt-BR'

    dataset = Dataset.from_pandas(df, split='train', features=Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
    }))

    return dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)


def push_dataset(dataset):
    dataset.push_to_hub(f'brazilian_literature', token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    files = readfiles()

    dataset = build_dataset(files)
    
    push_dataset(dataset)
