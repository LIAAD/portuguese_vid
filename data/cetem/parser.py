from pathlib import Path
import dotenv
import os
import re
from bs4 import BeautifulSoup
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict, concatenate_datasets
from tqdm import tqdm

dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent


def prepare_data():
    data_path = os.path.join(CURRENT_PATH, "data")

    for file in ['cetem_folha.txt', 'cetem_publico.txt']:
        with open(os.path.join(data_path, file), 'r', encoding='ISO-8859-1') as f, open(os.path.join(data_path, f"{file.replace('.txt','')}_utf8.txt"), 'w', encoding='utf-8') as w:
            data = f.read()

            data = data.strip()

            data = data.replace('\n', ' ')
            data = data.replace('<t>', ' ')
            data = data.replace('</t>', ' ')
            data = re.sub(r'\s+', ' ', data)
            data = data.replace('</ext>', '</ext>\n')

            w.write(data)


def process_line(line):
    line = line.strip()
    soap = BeautifulSoup(line, 'lxml')

    text = soap.get_text().replace('\n', ' ').replace(
        '\t', ' ').replace('\r', ' ').replace('  ', ' ').strip()
    text = re.sub(r'\s+', ' ', text)

    return text


def parse_correct_data():
    data_path = os.path.join(CURRENT_PATH, "data")

    final_dataset = []

    for variant_dict in [{'label': 'pt-PT', 'file': 'cetem_publico.txt'}, {'label': 'pt-BR', 'file': 'cetem_folha.txt'}]:
        dataset = []

        with open(os.path.join(data_path, variant_dict['file']), "r", encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines):
            dataset.append(process_line(line))

        df = pd.DataFrame(dataset, columns=['text'])
        df['label'] = variant_dict['label']

        dataset = Dataset.from_pandas(df, split='train', features=Features({
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
        }))

        final_dataset.append(dataset)

    dataset = concatenate_datasets(
        [dataset for dataset in final_dataset], split='train')

    dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
        'cetem', token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    #prepare_data()
    parse_correct_data()
