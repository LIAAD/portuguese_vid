from datasets import load_dataset
from datasets import Dataset, Features, Value, ClassLabel
import dotenv
from tqdm import tqdm
import os
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent


def readfiles():
    corpora = []
    data_path = os.path.join(CURRENT_PATH, 'data')

    for root, directories, files in os.walk(data_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding="utf-8") as f:
                    corpora.append(f.read().strip(). replace('\n', ' '))

    return corpora


def process_document(files):
    parsed_files = []

    for file in tqdm(files):

        for pattern in [r"<CHAPTER ID[^>]*>", r"<SPEAKER ID[^>]*>", r"<P*>"]:
            file = re.sub(pattern, "", file)

        file = file.strip()
        file = re.sub(r"\s+", " ", file)
        file = file.replace('\n', ' ')

        parsed_files.append(file)

    return parsed_files


if __name__ == "__main__":
    files = readfiles()

    files = process_document(files)

    df = pd.DataFrame(files, columns=['text'])

    df['label'] = 'pt-PT'

    dataset = Dataset.from_pandas(df, split='train', features=Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
    }))

    dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
        f'portuguese_europarl', token=os.getenv("HF_TOKEN"))