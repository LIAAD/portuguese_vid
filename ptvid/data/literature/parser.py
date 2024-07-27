import os

import bs4 as bs
import pandas as pd
from datasets import Dataset, concatenate_datasets
from nltk.tokenize import sent_tokenize
from pt_pump_up_admin.integrations.HuggingFace import HuggingFaceDataset
from tqdm import tqdm

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def lt_corpus():
    list_br_ids = ["L0639", "L0292", "L0418", "L0635", "L0476", "L0592", "L0092", "L0248", "L0380"]

    brazilian_texts = []
    portuguese_texts = []

    with open(os.path.join(CURRENT_PATH, "data", "LTCorpus.txt"), "r", encoding="utf-8") as f:
        xml_content = f.read()
        soup = bs.BeautifulSoup(xml_content, "lxml")
        texts = soup.find_all("text")

        for text in texts:
            if text["id"] in list_br_ids:
                brazilian_texts.append(text.get_text())
            else:
                portuguese_texts.append(text.get_text())

    # Create Dataframe where brazilian texts are labeled as 1 and portuguese texts are labeled as 0
    df = pd.DataFrame(
        {"text": brazilian_texts + portuguese_texts, "label": [1] * len(brazilian_texts) + [0] * len(portuguese_texts)}
    )

    return Dataset.from_pandas(df)


def kaggle_corpus():
    brazilian_texts = []

    # Iterate all .txt files in the folder
    data_folder = os.path.join(CURRENT_PATH, "data", "kaggle")

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), "r", encoding="ISO-8859-1") as f:
                brazilian_texts.append(f.read())

    # Create Dataframe where brazilian texts are labeled as 1 and portuguese texts are labeled as 0
    df = pd.DataFrame({"text": brazilian_texts, "label": [1] * len(brazilian_texts)})

    return Dataset.from_pandas(df)


def gutemberg_project():
    def pre_process_text(text):
        # Find the start and end of the actual text
        start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

        # Remove the header and footer
        return text[start:end]

    portuguese_texts = []

    data_folder = os.path.join(CURRENT_PATH, "data", "gutemberg")

    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
                portuguese_texts.append(pre_process_text(f.read()))

    # Create Dataframe where brazilian texts are labeled as 1 and portuguese texts are labeled as 0
    df = pd.DataFrame({"text": portuguese_texts, "label": [0] * len(portuguese_texts)})

    return Dataset.from_pandas(df)


def sentence_tokenizer(dataset: Dataset):
    new_dataset = {
        "text": [],
        "label": [],
    }

    for row in tqdm(dataset):
        sentences = sent_tokenize(row["text"], language="portuguese")

        # Each 4 sentences, add a new row to the dataset. Reject the first 2 sentences
        for i in range(2, len(sentences), 4):
            new_dataset["text"].append(" ".join(sentences[i : i + 4]))
            new_dataset["label"].append(row["label"])

    return Dataset.from_dict(new_dataset)


def parse():
    lt_dataset = lt_corpus()
    kaggle_dataset = kaggle_corpus()
    gutember_dataset = gutemberg_project()

    # Concatenate datasets
    all_dataset = concatenate_datasets([lt_dataset, kaggle_dataset, gutember_dataset])

    all_dataset = sentence_tokenizer(all_dataset)

    all_dataset_pt = all_dataset.filter(lambda example: example["label"] == 0)
    all_dataset_br = all_dataset.filter(lambda example: example["label"] == 1)

    hf_dataset = HuggingFaceDataset("", "text", False)

    hf_dataset.dataset = all_dataset_pt

    number_of_rows, number_tokens, number_sentences, number_characters = hf_dataset.produce_stats()

    print("Portuguese dataset stats:")
    print(f"Number of rows: {number_of_rows}")
    print(f"Number of tokens: {number_tokens}")
    print(f"Number of sentences: {number_sentences}")
    print(f"Number of characters: {number_characters}")

    hf_dataset.dataset = all_dataset_br

    number_of_rows, number_tokens, number_sentences, number_characters = hf_dataset.produce_stats()

    print("Brazilian dataset stats:")
    print(f"Number of rows: {number_of_rows}")
    print(f"Number of tokens: {number_tokens}")
    print(f"Number of sentences: {number_sentences}")
    print(f"Number of characters: {number_characters}")


def load_hf(dataset_name):
    lt_dataset = lt_corpus()
    kaggle_dataset = kaggle_corpus()
    gutember_dataset = gutemberg_project()

    # Concatenate datasets
    all_dataset = concatenate_datasets([lt_dataset, kaggle_dataset, gutember_dataset])

    all_dataset = sentence_tokenizer(all_dataset)

    all_dataset.shuffle()

    all_dataset.push_to_hub(
        dataset_name,
        config_name="literature",
    )


if __name__ == "__main__":
    parse()
