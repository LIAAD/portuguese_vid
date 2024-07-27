from bs4 import BeautifulSoup
import requests
from datasets import Dataset, Features, Value, ClassLabel
import pandas as pd
import dotenv
import os
import time
from tqdm import tqdm

dotenv.load_dotenv(dotenv.find_dotenv())

prefix = 'https://www.camara.leg.br/internet/sitaqweb'


"https://www.camara.leg.br/internet/sitaqweb/resultadoPesquisaDiscursos.asp?txIndexacao=&CurrentPage=2&BasePesq=plenario&txOrador=&txPartido=&dtInicio=01/01/2021&dtFim=08/08/2023&txUF=&txSessao=&listaTipoSessao=&listaTipoInterv=&inFalaPres=&listaTipoFala=&listaFaseSessao=&txAparteante=&listaEtapa=&CampoOrdenacao=dtSessao&TipoOrdenacao=DESC&PageSize=50&txTexto=&txSumario="


speeches = []

for page in range(1, 100):

    print(f"Page {page}")

    html_page = requests.get(
        f"{prefix}/resultadoPesquisaDiscursos.asp?CurrentPage={page}&BasePesq=plenario&txIndexacao=&txOrador=&txPartido=&dtInicio=01/01/2021&dtFim=08/08/2023&txUF=&txSessao=&listaTipoSessao=&listaTipoInterv=&inFalaPres=&listaTipoFala=&listaFaseSessao=&txAparteante=&listaEtapa=&CampoOrdenacao=dtSessao&TipoOrdenacao=DESC&PageSize=50&txTexto=&txSumario="
    )

    if html_page.status_code != 200:
        print(f"Page {page} not found")
        break

    root = BeautifulSoup(html_page.content, 'html.parser')

    for link in tqdm(root.find_all('a', title='Íntegra do Discurso')):

        correct_link = link.get('href').replace('\r', '').replace(
            '\n', '').replace('\t', '').replace(' ', '%20').strip()

        speech = requests.get(f"{prefix}/{correct_link}")

        if speech.status_code != 200:
            print(f"Speech {correct_link} not found")
            continue

        soap = BeautifulSoup(speech.content.decode(
            'utf-8', 'ignore'), 'html.parser')

        beautified_text = soap.getText().replace(
            '\r', '').replace('\n', '').replace('\t', '').strip()

        index_of_speaker = beautified_text.find('O SR.')

        if index_of_speaker == -1:
            index_of_speaker = beautified_text.find('A SRA.')

        if index_of_speaker == -1:
            print(f"Speech {correct_link} not found")
            continue

        beautified_text = beautified_text[index_of_speaker:]

        speeches.append(beautified_text)

        #time.sleep(1)

df = pd.DataFrame(speeches, columns=['text'])

df['label'] = 'pt-BR'

dataset = Dataset.from_pandas(df, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
    'brazilian_senate_speeches_1', token=os.getenv('HF_TOKEN'))
