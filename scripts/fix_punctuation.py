"""
Script used to merge the different sources of the PtBRVId corpus.
"""

import logging
import multiprocessing as mp
import re
from typing import List

from vllm import LLM, SamplingParams
import datasets
import justext
import numpy as np
from cleantext import clean
from nltk.tokenize import word_tokenize
import torch

from ptvid.constants import DOMAINS

logging.basicConfig(level=logging.INFO)


N = 500
PROMPT = """I have a dataset in Portuguese that contains punctuation errors.
Your task is to correct these errors.
I will provide you with the text containing the errors, and you will return the corrected version.

# Text with Errors:
{text}

# Corrected Text:
"""

def train_valid_split(dataset):
    label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
    valid_label0 = label0.select(range(N))
    train_label0 = label0.select(range(N, len(label0)))
    label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
    valid_label1 = label1.select(range(N))
    train_label1 = label1.select(range(N, len(label1)))

    trainset = datasets.concatenate_datasets([train_label0, train_label1])
    validset = datasets.concatenate_datasets([valid_label0, valid_label1])

    return trainset, validset


def main(
    raw_dataset_name: str = "liaad/PtBrVId",
    clean_dataset_name: str = "liaad/PtBrVId_fix",
):  
    model = LLM("meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=torch.cuda.device_count())
    params = SamplingParams(max_tokens=512)
    for domain in ["literature", "journalistic"]:
        logging.info("loading dataset")
        dataset = datasets.load_dataset(raw_dataset_name, domain, split="train")
        dataset = dataset.select(range(100))
        dataset = dataset.map(lambda x: {"prompt": PROMPT.format(text=x["text"])}, num_proc=mp.cpu_count())
        outputs = model.generate(dataset["prompt"], params)
        completions = [output.outputs[0].text for output in outputs]
        dataset = dataset.remove_columns(["prompt"])
        dataset = dataset.add_column("text_fix", completions)

        if domain == "web":
            dataset = dataset.remove_columns("domain")

        logging.info("push to hub")
        dataset.push_to_hub(clean_dataset_name, domain, split="train")


if __name__ == "__main__":
    main()
    