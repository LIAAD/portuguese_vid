from datasets import load_dataset, Dataset, DatasetDict
import os
import pandas as pd
from urllib.parse import urlparse
import time
from concurrent.futures import ThreadPoolExecutor

# Record the time elapsed
start_time = time.time()

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def save_to_csv(outputs):
    brazil_df = pd.DataFrame(outputs['brazil'])
    portugal_df = pd.DataFrame(outputs['portugal'])

    # Group by subdomains and count
    brazil_df = brazil_df.groupby(
        ['sub_domains']).size().reset_index(name='count')
    portugal_df = portugal_df.groupby(
        ['sub_domains']).size().reset_index(name='count')

    # Sort by count
    brazil_df = brazil_df.sort_values(by=['count'], ascending=False)
    portugal_df = portugal_df.sort_values(by=['count'], ascending=False)

    brazil_df.to_csv(os.path.join(CURRENT_PATH, 'out', 'brazil.csv'))
    portugal_df.to_csv(os.path.join(CURRENT_PATH, 'out', 'portugal.csv'))

    print(f"Number of subdomains in Brazil: {len(brazil_df)}")
    print(f"Number of subdomains in Portugal: {len(portugal_df)}")


def frequency_table():
    oscar_dataset = load_dataset(
        'oscar-corpus/OSCAR-2301', 'pt', streaming=True, split='train')

    outputs = {
        'portugal': {
            'sub_domains': []
        },
        'brazil': {
            'sub_domains': []
        },
    }

    sub_domains = []

    for idx, row in enumerate(oscar_dataset):

        if idx % 10_000 == 0:
            print(f"Processed {idx} rows")
            save_to_csv(outputs)

        # Remove top level domain from uri

        domains = urlparse(row['meta']['warc_headers']
                           ['warc-target-uri']).netloc.split('.')

        top_level_domain = domains[-1]

        if len(domains) > 2:
            sub_domains = '.'.join(domains[1:])
        else:
            continue

        if top_level_domain == 'pt':
            outputs['portugal']['sub_domains'].append(sub_domains)

        elif top_level_domain == 'br':
            outputs['brazil']['sub_domains'].append(sub_domains)

    save_to_csv(outputs)

    # Print the minutes and seconds elapsed
    print(f"Time elapsed: {(time.time() - start_time) / 60} minutes")


def parse():
    with open(os.path.join(CURRENT_PATH, 'data', 'portugal.txt'), 'r') as file:
        # Create a set of portuguese domains
        portugal_domains = set([domain.strip() for domain in file.readlines()])

        def process_portuguese(domain):

            for portugal_domain in portugal_domains:
                if portugal_domain in domain:
                    return True

            return False

    with open(os.path.join(CURRENT_PATH, 'data', 'brazil.txt'), 'r') as file:
        brazil_domains = set([domain.strip() for domain in file.readlines()])

        def process_brazilian(domain):

            for brazil_domain in brazil_domains:
                if brazil_domain in domain:
                    return True

            return False

    def parse_thread(process_strategy, filename):

        df = pd.DataFrame(columns=['text', 'domain'])

        print(f"Processing {filename}")

        oscar_dataset = load_dataset(
            'oscar-corpus/OSCAR-2301', 'pt', streaming=True, split='train')

        list_results = {
            'text': [],
            'domain': []
        }

        for idx, row in enumerate(oscar_dataset):

            if idx % 10_000 == 0 and idx != 0:
                print(f"Processed {idx} rows | filename: {filename}")
                print(f"Finished {
                      len(list_results['text'])} rows | filename: {filename}")

            if len(list_results['text']) >= 100_000:
                print(f"Finished {
                      len(list_results['text'])} rows | filename: {filename}")
                df = pd.concat([df, pd.DataFrame(list_results)],
                               ignore_index=True)
                df.to_csv(os.path.join(CURRENT_PATH, 'out', filename))
                break

            domain = urlparse(row['meta']['warc_headers']
                              ['warc-target-uri']).netloc

            if process_strategy(domain):
                list_results['text'].append(row['text'])
                list_results['domain'].append(domain)

        return df

    with ThreadPoolExecutor() as executor:
        portuguese_data = executor.submit(
            parse_thread, process_portuguese, 'portugal.csv')
        brazilian_data = executor.submit(
            parse_thread, process_brazilian, 'brazil.csv')

    df_pt = portuguese_data.result()
    df_br = brazilian_data.result()

    df_pt.to_csv(os.path.join(CURRENT_PATH, 'out',
                 'portugal.csv'), encoding='utf-8')
    df_br.to_csv(os.path.join(CURRENT_PATH, 'out',
                 'brazil.csv'), encoding='utf-8')


def load_hf(dataset_name):
    df_br = pd.read_csv(os.path.join(CURRENT_PATH, 'out',
                        'brazil.csv'), encoding='utf-8')
    df_pt = pd.read_csv(os.path.join(CURRENT_PATH, 'out',
                        'portugal.csv'), encoding='utf-8')

    # Create Column label for brazilian data
    df_br['label'] = 1
    df_pt['label'] = 0

    # Concatenate both dataframes
    df = pd.concat([df_br, df_pt], ignore_index=True)

    # Remove Column Unnamed: 0
    df = df.drop(columns=['Unnamed: 0'])

    dataset = Dataset.from_pandas(df)

    # Shuffle the dataset
    dataset = dataset.shuffle()

    dataset.push_to_hub(
        dataset_name,
        config_name='web',
        private=False
    )
