from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import os
from multiprocessing import Pool
from pathlib import Path
from process_domains import process_domain
from ftlangdetect import detect
from cleantext import clean
import numpy as np
import logging

CURRENT_PATH = Path(__file__).parent

logging.basicConfig(level=logging.DEBUG, filename=os.path.join(CURRENT_PATH, 'out',
                    'build_dataset.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')

FAST_TEXT_THRESHOLD = 0.7


def load_data():
    return [
        {
            "domain": "politics",
            "dataset": load_dataset("arubenruben/portuguese_europarl"),
        },
        {
            "domain": "politics",
            "dataset": load_dataset("arubenruben/brazilian_senate_speeches"),
        },
        {
            "domain": "literature",
            "dataset": load_dataset("arubenruben/arquivo-pessoa"),
        },
        {
            "domain": "literature",
            "dataset": load_dataset("arubenruben/gutemberg-portuguese-novels"),
        },
        {
            "domain": "literature",
            "dataset": load_dataset("arubenruben/brazilian_literature")
        },
        {
            "domain": "web",
            "dataset": load_dataset("arubenruben/portuguese-oscar-li"),
        },
        {
            "domain": "news",
            "dataset": load_dataset("arubenruben/cetem"),
        },
        {
            "domain": "law",
            "dataset": load_dataset("arubenruben/lener_br_language_id"),
        },
        {
            "domain": "law",
            "dataset": load_dataset("arubenruben/portuguese_legal_sentences"),
        },
        {
            "domain": "news",
            "dataset": load_dataset("arubenruben/portuguese_dslcc"),
        },
        {
            "domain": "literature",
            "dataset": load_dataset("arubenruben/LT-Corpus"),
        },
        {
            "domain": "social_media",
            "dataset": load_dataset("arubenruben/hate_br_li"),
        },
        {
            "domain": "social_media",
            "dataset": load_dataset("arubenruben/fake_news_wpp_br"),
        },
        {
            "domain": "social_media",
            "dataset": load_dataset("arubenruben/portuguese_presidential_elections_tweets_lid"),
        }
    ]


def clean_dataset(dataset):

    def clean_nan(df_dataset):
        logging.info(
            f"Number of rows with NaN values: {df_dataset.isnull().sum()}")
        return df_dataset.dropna()

    def clean_empty_strings(df_dataset):
        logging.info(
            f"Number of rows with empty strings: {len(df_dataset[df_dataset['text'] == ''])}")
        return df_dataset[df_dataset['text'] != '']

    def clean_duplicates(df_dataset):
        logging.info(
            f"Number of rows with duplicated text: {len(df_dataset[df_dataset.duplicated(subset=['text'])])}")

        return df_dataset.drop_duplicates(subset=['text'])

    def clean_non_portuguese(df_dataset):
        indexes_to_drop = []

        for index, row in df_dataset.iterrows():
            if index % 10000 == 0 and index > 0:
                logging.info(f"Processing row {index} of {len(df_dataset)}")

            detected_lang = detect(row['text'])

            if detected_lang['lang'] != 'pt' or detected_lang['score'] < FAST_TEXT_THRESHOLD:
                indexes_to_drop.append(index)

        logging.info(
            f"Number of rows with non-portuguese text: {len(indexes_to_drop)}")

        df = df_dataset.drop(indexes_to_drop)

        return df

    def clean_text(df_dataset):

        for index, row in df_dataset.iterrows():

            if index % 10000 == 0 and index > 0:
                logging.info(f"Processing row {index} of {len(df_dataset)}")

            df_dataset.at[index, 'text'] = clean(row['text'], fix_unicode=True, to_ascii=False, lower=False, no_line_breaks=True, no_urls=True,
                                                 no_emails=True, no_phone_numbers=True, no_numbers=True, no_digits=True,  no_currency_symbols=True, no_punct=False, lang="en")

        return df_dataset

    def remove_outliers(df_dataset):
        tokens_count = df_dataset['text'].apply(lambda x: len(x.split()))

        # Mean LogNormal Distribution
        mean = np.log(tokens_count).mean()
        std = np.log(tokens_count).std()

        # Remove All Row with token count > 3 * std
        indexes_to_drop = []

        for index, row in df_dataset.iterrows():
            if index % 10000 == 0 and index > 0:
                logging.info(f"Processing row {index} of {len(df_dataset)}")

            if len(row['text'].split()) > np.exp(mean + 3 * std):
                indexes_to_drop.append(index)

        logging.info(f"Number of rows with outliers: {len(indexes_to_drop)}")

        df_dataset = df_dataset.drop(indexes_to_drop)

        return df_dataset

    df = dataset.to_pandas()

    logging.info(f"Original Number of rows: {len(dataset)}")

    df = clean_nan(df)

    df = clean_empty_strings(df)

    df = clean_duplicates(df)

    df = clean_non_portuguese(df)

    df = clean_text(df)

    df = remove_outliers(df)

    df.reset_index(drop=True, inplace=True)

    logging.info(f"Final Number of rows: {len(df)}")

    return Dataset.from_pandas(df, split='train', features=dataset.features)


def process(domain_dataset):
    domain = domain_dataset[0]

    logging.info(f"Processing domain {domain}")

    dataset = concatenate_datasets([dataset for dataset in domain_dataset[1]])

    dataset = clean_dataset(dataset)

    dataset_dict = process_domain(dataset, domain)

    dataset_dict['train'].to_csv(os.path.join(
        CURRENT_PATH, "data", domain, "train.csv"))

    dataset_dict['validation'].to_csv(os.path.join(
        CURRENT_PATH, "data", domain, "validation.csv"))

    dataset_dict['test'].to_csv(os.path.join(
        CURRENT_PATH, "data", domain, "test.csv"))


def main():
    datasets = load_data()

    final_dataset = {}

    for dataset in datasets:
        domain = dataset["domain"]

        dataset = dataset["dataset"]

        if 'validation' in dataset.keys():
            dataset = concatenate_datasets(
                [dataset["train"], dataset["validation"], dataset["test"]])
        else:
            dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        if domain not in final_dataset.keys():
            final_dataset[domain] = [dataset]
        else:
            final_dataset[domain].append(dataset)

    domain_datasets = [(domain, datasets)
                       for domain, datasets in final_dataset.items()]

    # Define the number of processes you want to run in parallel
    num_processes = min(len(domain_datasets), os.cpu_count())

    # Create a pool of processes
    with Pool(num_processes) as pool:
        pool.map(process, domain_datasets)


if __name__ == '__main__':
    main()
