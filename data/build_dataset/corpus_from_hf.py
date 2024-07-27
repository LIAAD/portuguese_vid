from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import os
import logging
from process_domains import process_domain


CURRENT_PATH = Path(__file__).parent

logging.basicConfig(level=logging.DEBUG, filename=os.path.join(CURRENT_PATH, 'out',
                    'build_dataset.log'), filemode='w', format='%(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    for domain in ['law']:
        
        dataset = load_dataset(
            "arubenruben/Portuguese_Language_Identification", domain)
        
        dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

        dataset_dict = process_domain(dataset=dataset, domain=domain, ploting=False)

        dataset_dict['train'].to_csv(os.path.join(CURRENT_PATH, "data", domain, "train.csv"))

        dataset_dict['validation'].to_csv(os.path.join(CURRENT_PATH, "data", domain, "validation.csv"))

        dataset_dict['test'].to_csv(os.path.join(CURRENT_PATH, "data", domain, "test.csv"))

