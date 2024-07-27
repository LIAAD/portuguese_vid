from pathlib import Path
import dotenv
import os 
from datasets import Dataset, Features, Value, ClassLabel
from pypdf import PdfReader
from tqdm import tqdm

dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent

DATA_PATH = os.path.join(CURRENT_PATH, "data")

corpora = {
    'text': [],
    'label': []
}

def process_txt(file):
    with open(file, 'r', encoding="utf-8") as f:
        string = f.read().strip(). replace('\n', ' ')

        # Remove 10% of the text at the beginning and at the end
        string = string[int(len(string)*0.1):int(len(string)*0.9)]

        return string

def process_pdf(file):
    corpus = ""

    for page in tqdm(PdfReader(file).pages):
        string = page.extract_text()

        string = string.strip().replace('\n', ' ')

        corpus += string

        print(f"Corpus size: {len(corpus)}")

    return corpus

for root, directories, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.txt'):
            string = process_txt(os.path.join(root, file))
                    
        corpora['text'].append(string)
        corpora['label'].append('pt-PT')

dataset = Dataset.from_dict(corpora, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).push_to_hub('gutemberg-portuguese-novels', token=os.getenv('HF_TOKEN'))

