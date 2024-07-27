from bs4 import BeautifulSoup
from pathlib import Path
import os
from datasets import Dataset, Features, Value, ClassLabel
import dotenv
import re

dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent

list_brazilian_files = [
    'L0476',
    'L0592',
    'L0092',
    'L0639',
    'L0292',
    'L0418',
    'L0635',
    'L0248',
    'L0380',
]

dataset = {
    'text': [],
    'label': []
}


def beautify_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    return text


with open(os.path.join(CURRENT_PATH, 'data', 'LTCorpus.txt'), 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'lxml')

    for text in soup.find_all('text'):
        id = text['id']
        text = beautify_text(text.get_text())

        dataset['text'].append(text)

        if id in list_brazilian_files:
            dataset['label'].append('pt-BR')
        else:
            dataset['label'].append('pt-PT')

dataset = Dataset.from_dict(dataset, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).push_to_hub(
    'LT-Corpus', token=os.getenv("HF_TOKEN"), private=True
)
