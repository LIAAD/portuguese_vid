"""
Script used to merge the different sources of the PtBRVId corpus.
"""

import logging
import multiprocessing as mp
import re
from typing import List

import datasets
import justext
import numpy as np
from cleantext import clean
from nltk.tokenize import word_tokenize

from ptvid.constants import DOMAINS

logging.basicConfig(level=logging.INFO)

N = 500


class PortugueseDetokenizer:
    """Based on the TreebankWordDetokenizer from nltk."""

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r"(\S)\s(\'\')"), r"\1\2"),
        (re.compile(r"(\S)\s(»)"), r"\1\2"),
        (re.compile(r"(\'\')\s([.,:)\]>};%])"), r"\1\2"),
        (re.compile(r"''"), '"'),
    ]

    # Undo padding on parentheses.
    PARENS_BRACKETS = [
        (re.compile(r"([\[\(\{\<])\s"), r"\g<1>"),
        (re.compile(r"\s([\]\)\}\>])"), r"\g<1>"),
        (re.compile(r"([\]\)\}\>])\s([:;,.])"), r"\1\2"),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r"([^'])\s'\s"), r"\1' "),
        (re.compile(r"\s([?!.])"), r"\g<1>"),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r"\1\2\3"),
        (re.compile(r"([#$])\s"), r"\g<1>"),
        (re.compile(r"\s([;%])"), r"\g<1>"),
        (re.compile(r"\s\.\.\.\s"), r"..."),
        (re.compile(r"\s([:,])"), r"\1"),
    ]

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r"([ (\[{<])\s``"), r"\1``"),
        (re.compile(r"(``)\s"), r"\1"),
        (re.compile(r"(`)\s"), r"\1"),
        (re.compile(r"(«)\s"), r"\1"),
        (re.compile(r"``"), r'"'),
    ]

    PRONOUNS = [
        " -me", 
        " -te", 
        " -se", 
        " -nos", 
        " -vos", 
        " -o", 
        " -a", 
        " -os", 
        " -as", 
        " -lhe", 
        " -lhes", 
        " -lho", 
        " -lha", 
        " -lhos", 
        " -lhas"
    ]

    # Regex to remove space before hyphen if it is connected to a pronoun
    PRONOUNS_REGEX = re.compile(r"(\S)\s(-" + "|".join(PRONOUNS) + r")")

    def detokenize(self, tokens: List[str]) -> str:
        """Duck-typing the abstract *tokenize()*."""
        text = " ".join(tokens)

        # Add extra space to make things easier
        text = " " + text + " "

        # Reverse the regexes applied for ending quotes.
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        # Undo the space padding.
        text = text.strip()

        text = regexp.sub(substitution, text)

        # Reverse the padding regexes applied for parenthesis/brackets.
        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for punctuations.
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for starting quotes.
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for pronoun in self.PRONOUNS:
            text = text.replace(pronoun, pronoun.strip())

        return text.strip()


def clean_web(dataset):
    def _clean_text(example):
        paragraphs = justext.justext(example["text"], justext.get_stoplist("Portuguese"))
        text = " ".join(paragraph.text for paragraph in paragraphs if paragraph.class_type == "good")
        return {"text": text}

    dataset = dataset.map(_clean_text, num_proc=mp.cpu_count())
    dataset = dataset.filter(lambda x: x["text"] != "", num_proc=mp.cpu_count())
    return dataset


def clean_text(dataset):
    def _clean_text(example):
        text = clean(
            example["text"],
            fix_unicode=True,
            to_ascii=True,
            lower=False,
            normalize_whitespace=True,
        )
        return {"text": text}

    dataset = dataset.map(_clean_text, num_proc=mp.cpu_count())
    dataset = dataset.filter(lambda x: x["text"] != "", num_proc=mp.cpu_count())
    return dataset


def drop_outliers(dataset):
    def count_tokens(example):
        tokens = word_tokenize(example["text"], "portuguese")
        return {"n_tokens": len(tokens)}

    dataset = dataset.map(count_tokens, num_proc=mp.cpu_count())
    q1 = np.percentile(dataset["n_tokens"], 25)
    q3 = np.percentile(dataset["n_tokens"], 75)
    iqr = q3 - q1
    min_tokens = q1 - 1.5 * iqr
    max_tokens = q3 + 1.5 * iqr
    dataset = dataset.filter(lambda x: min_tokens < x["n_tokens"] < max_tokens, num_proc=mp.cpu_count())
    dataset = dataset.remove_columns("n_tokens")
    return dataset


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


def fix_tokens(dataset):
    detokenizer = PortugueseDetokenizer()

    def _fix_tokens(example):
        tokens = example["text"].split(" ")
        text = detokenizer.detokenize(tokens)
        return {"text": text}

    dataset = dataset.map(_fix_tokens, num_proc=mp.cpu_count())
    return dataset


def drop_duplicates(dataset):
    df = dataset.to_pandas()
    df = df.drop_duplicates(ignore_index=True)
    dataset = datasets.Dataset.from_pandas(df)
    return dataset


def main(
    raw_dataset_name: str = "arubenruben/portuguese-language-identification-raw",
    clean_dataset_name: str = "liaad/PtBrVId",
):
    for domain in DOMAINS:
        logging.info("loading dataset")
        dataset = datasets.load_dataset(raw_dataset_name, domain, split="train")

        if domain == "web":
            dataset = dataset.remove_columns("domain")

        logging.info("drop empty")
        dataset = dataset.filter(lambda x: x["text"] != "", num_proc=mp.cpu_count())

        logging.info("drop duplicates")
        dataset = drop_duplicates(dataset)

        logging.info("clean text")
        dataset = clean_text(dataset)

        logging.info("fix tokens")
        if domain in ["literature", "journalistic"]:
            dataset = fix_tokens(dataset)

        logging.info("clean web")
        if domain == "web":
            dataset = clean_web(dataset)

        logging.info("drop outliers")
        dataset = drop_outliers(dataset)

        logging.info("split into train and valid")
        trainset, validset = train_valid_split(dataset)

        logging.info("push to hub")
        trainset.push_to_hub(clean_dataset_name, domain, split="train")
        validset.push_to_hub(clean_dataset_name, domain, split="valid")


if __name__ == "__main__":
    main()
    