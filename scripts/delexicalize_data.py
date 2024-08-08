"""Cache the delexalized data for training."""

import logging

import numpy as np

from ptvid.constants import DATASET_NAME, DOMAINS, SAMPLE_SIZE
from ptvid.src.data import Data

logging.basicConfig(level=logging.INFO)


def main(dataset_name: str):
    train_data = Data(dataset_name, split="train")
    valid_data = Data(dataset_name, split="valid")

    start_pos_prob = 0.0
    stop_pos_prob = 1.0
    for pos_prob in np.arange(start_pos_prob, stop_pos_prob + 0.1, 0.1):
        for ner_prob in np.arange(0.0, 1.0 + 0.1, 0.1):
            for domain in DOMAINS:
                logging.info(f"Running {domain} pos_prob={pos_prob}, ner_prob={ner_prob}")

                logging.info("Loading train data.")
                train_loader = train_data.load_domain(
                    domain, balance=True, pos_prob=pos_prob, ner_prob=ner_prob, sample_size=SAMPLE_SIZE
                )
                logging.info(f"Resulting dataset {train_loader}")

                logging.info("Loading valid data.")
                valid_loader = valid_data.load_domain(domain, balance=True)
                logging.info(f"Resulting dataset {valid_loader}")


if __name__ == "__main__":
    main(dataset_name=DATASET_NAME)
