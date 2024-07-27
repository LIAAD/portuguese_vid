import os
from bs4 import BeautifulSoup
from datasets import Dataset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_xml():
    # Read all .txt files in the europarl directory

    for country_code in tqdm(os.listdir(os.path.join(CURRENT_DIR, 'data'))):

        results = {
            'speaker': [],
            'text': [],
        }

        for filename in tqdm(os.listdir(os.path.join(CURRENT_DIR, 'data', country_code))):
            if filename.endswith('.xml'):
                with open(os.path.join(CURRENT_DIR, 'data', country_code, filename), 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'xml')
                    for intervention in soup.find_all('intervention'):
                        speaker = intervention.find('speaker').text
                        speech = intervention.find('speech').text
                        results['speaker'].append(speaker)
                        results['text'].append(speech)

        dataset = Dataset.from_dict(results, split="train")
        
        dataset.shuffle()
        
        dataset.push_to_hub(
            'arubenruben/europarl',
            config_name=country_code,
        )

parse_xml()