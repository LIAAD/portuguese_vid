from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from datasets import Dataset, Features, Value, ClassLabel
import os
import dotenv
import time

dotenv.load_dotenv(dotenv.find_dotenv())

def process_page(soup):
    content = soup.find('div', {'class': 'content'})
    
    text = content.find('div', {'class': 'texto-poesia'})
    
    if text is None:
        text = content.find('div', {'class': 'texto-prosa'})
    
    if text is None:
        print("Error: Text not found")
        return None

    return text.get_text().strip().replace('\n', ' ')

corpora = {
    'text': [],
    'label': []
}

for i in tqdm(range(4, 4544)):
    url = f'http://www.arquivopessoa.net/textos/{i}'
    
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {url}")
        continue
    
    soup = BeautifulSoup(response.content, 'html.parser')

    page = process_page(soup)

    corpora['text'].append(page)
    corpora['label'].append('pt-PT')
    
    time.sleep(0.5)

dataset = Dataset.from_dict(corpora, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).push_to_hub('arquivo-pessoa', token=os.getenv('HF_TOKEN'))

