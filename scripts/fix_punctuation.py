"""
Script used to merge the different sources of the PtBRVId corpus.
"""

import logging

import datasets

from ptvid.constants import N_PROC
from ptvid.data.detokenizer import PortugueseDetokenizer

logging.basicConfig(level=logging.INFO)


def fix_journalistic(dataset):
    detokenizer = PortugueseDetokenizer()

    label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=N_PROC)
    label0 = label0.map(lambda x: {"text": detokenizer.detokenize(x["text"].split())})

    label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=N_PROC)
    label1 = label1.map(lambda x: {"text": x["text"].replace(" .", ".")})

    dataset = datasets.concatenate_datasets([label0, label1])
    return dataset


def fix_literature(dataset):
    detokenizer = PortugueseDetokenizer()
    dataset = dataset.map(lambda x: {"text": detokenizer.detokenize(x["text"].split())})
    return dataset


def fix_legal(dataset):
    detokenizer = PortugueseDetokenizer()

    label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=N_PROC)

    label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=N_PROC)
    label1 = label1.map(lambda x: {"text": detokenizer.detokenize(x["text"].split())})

    dataset = datasets.concatenate_datasets([label0, label1])
    return dataset


def main(
    raw_dataset_name: str,
    clean_dataset_name: str,
):
    for domain in ["journalistic", "legal"]:
        for split in ["train", "valid"]:
            logging.info("loading dataset")
            dataset = datasets.load_dataset(raw_dataset_name, domain, split=split)

            if domain == "journalistic":
                dataset = fix_journalistic(dataset)

            if domain == "literature":
                dataset = fix_literature(dataset)

            if domain == "legal":
                dataset = fix_legal(dataset)

            logging.info("push to hub")
            dataset.push_to_hub(clean_dataset_name, domain, split=split)


if __name__ == "__main__":
    main()
