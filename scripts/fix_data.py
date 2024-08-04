"""
Script used to merge the different sources of the PtBRVId corpus.
"""

import json
import multiprocessing as mp
import re
from typing import List

import numpy as np
from nltk.tokenize import word_tokenize
import datasets
import justext
from cleantext import clean
from tqdm import tqdm

from ptvid.constants import DOMAINS, ROOT

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

        return text.strip()


def kind_jaccard_sim(str1, str2):
    dif = 0
    unique = set(str1) | set(str2)
    for char in unique:
        c1 = max(str1.count(char), 0)
        c2 = max(str2.count(char), 0)
        dif += abs(c1 - c2)
    return dif


def build_test():
    detokenizer = PortugueseDetokenizer()
    for domain in DOMAINS:
        print(domain)
        trainset = datasets.load_dataset("liaad/PtBrVId", domain)
        testset = datasets.load_dataset("LCA-PORVID/gold_labelled", domain)

        def process(example):
            text = example["text"]
            text = text.replace("<<NUMBER>>", "")
            text = text.replace("<<DIGIT>>", "")
            text = text.replace("<<CUR>>", "")
            text = text.replace("<<PHONE>>", "")
            text = text.replace("<<URL>>", "")
            text = text.replace("<<EMAIL>>", "")
            text = detokenizer.detokenize(text.split(" "))
            example["ptext"] = text
            return example

        testset = testset.map(process, num_proc=mp.cpu_count())

        scores, train_idxs, train_texts = [], [], []
        for sentence, label in tqdm(
            zip(testset["test"]["ptext"], testset["test"]["label"]), total=len(testset["test"])
        ):
            ref_text = sentence.replace(" ", "")

            def compute_distance(example):
                text = example["text"].replace(" ", "")
                example["score"] = kind_jaccard_sim(text, ref_text)
                return example

            trainset = trainset.map(compute_distance, num_proc=mp.cpu_count())
            min_score = min(trainset["train"]["score"])
            scores.append(min_score)
            min_idx = trainset["train"]["score"].index(min_score)
            train_idxs.append(min_idx)
            train_texts.append(trainset["train"]["text"][min_idx])

        testset["test"] = testset["test"].add_column("scores", scores)
        testset["test"] = testset["test"].add_column("train_idx", train_idxs)
        testset["test"] = testset["test"].add_column("train_text", train_texts)
        testset.push_to_hub("hugosousa/testset", domain)


def push_clean_test():
    detokenizer = PortugueseDetokenizer()
    for domain in DOMAINS:
        inpath = ROOT / "tmp" / f"{domain}.json"
        content = json.load(inpath.open())
        testset = datasets.Dataset.from_list(content)

        def process(example):
            if example["train_idx"] == -1:
                text = detokenizer.detokenize(example["text"].split(" "))
            else:
                text = example["train_text"]
            example["text"] = text
            return example

        testset = testset.map(process, num_proc=1)
        testset = testset.remove_columns("train_text")
        testset.push_to_hub("hugosousa/testset2", domain)


def manual_fix_testset():
    for domain in DOMAINS:
        dataset = datasets.load_dataset("hugosousa/testset", domain)
        df = dataset["test"].to_pandas()
        outfile = ROOT / "tmp" / f"{domain}.json"
        outfile.parent.mkdir(exist_ok=True)
        content = json.loads(df.to_json(orient="records"))
        json.dump(content, outfile.open("w"), indent=4, ensure_ascii=False)


def build_final():
    for domain in DOMAINS:
        dataset = datasets.load_dataset("liaad/PtBrVId", domain, split="train")
        testset = datasets.load_dataset("hugosousa/testset2", domain, split="train")

        idxs2drop = [idx for idx in testset["train_idx"] if idx != -1]
        idxs2keep = [i for i in range(len(dataset)) if i not in idxs2drop]
        dataset = dataset.select(
            idxs2keep,
        )

        testset = testset.remove_columns("train_idx")

        label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
        valid_label0 = label0.select(range(N))
        train_label0 = label0.select(range(N, len(label0)))
        label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
        valid_label1 = label1.select(range(N))
        train_label1 = label1.select(range(N, len(label1)))

        trainset = datasets.concatenate_datasets([train_label0, train_label1])
        validset = datasets.concatenate_datasets([valid_label0, valid_label1])
        new_dataset = datasets.DatasetDict(
            {
                "train": trainset,
                "valid": validset,
                "test": testset,
            }
        )
        new_dataset.push_to_hub("liaad/PtBrVId", domain)


def resize_valid():
    for domain in DOMAINS:
        trainset = datasets.load_dataset("liaad/PtBrVId", domain, split="train")
        validset = datasets.load_dataset("liaad/PtBrVId", domain, split="valid")

        label0 = validset.filter(lambda x: x["label"] == 0)  # , num_proc=mp.cpu_count())
        valid_label0 = label0.select(range(N))
        train_label0 = label0.select(range(N, len(label0)))
        label1 = validset.filter(lambda x: x["label"] == 1)  # , num_proc=mp.cpu_count())
        valid_label1 = label1.select(range(N))
        train_label1 = label1.select(range(N, len(label1)))

        trainset = datasets.concatenate_datasets([trainset, train_label0, train_label1])
        validset = datasets.concatenate_datasets([valid_label0, valid_label1])

        trainset.push_to_hub("liaad/PtBrVId", domain, split="train")
        validset.push_to_hub("liaad/PtBrVId", domain, split="valid")


def clean_web():
    trainset = datasets.load_dataset("liaad/PtBrVId", "web", split="train")
    validset = datasets.load_dataset("liaad/PtBrVId", "web", split="valid")
    dataset = datasets.concatenate_datasets([trainset, validset])

    def clean_text(example):
        paragraphs = justext.justext(example["text"], justext.get_stoplist("Portuguese"))
        text = " ".join(paragraph.text for paragraph in paragraphs if paragraph.class_type == "good")
        return {"text": text}

    dataset = dataset.map(clean_text, num_proc=mp.cpu_count())
    dataset = dataset.filter(lambda x: x["text"] != "", num_proc=mp.cpu_count())

    # add new valid examples
    label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
    valid_label0 = label0.select(range(N))
    train_label0 = label0.select(range(N, len(label0)))
    label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
    valid_label1 = label1.select(range(N))
    train_label1 = label1.select(range(N, len(label1)))

    trainset = datasets.concatenate_datasets([train_label0, train_label1])
    validset = datasets.concatenate_datasets([valid_label0, valid_label1])

    trainset.push_to_hub("liaad/PtBrVId", "web", split="train")
    validset.push_to_hub("liaad/PtBrVId", "web", split="valid")


def clean_text():
    def clean_text(example):
        text = clean(
            example["text"],
            fix_unicode=True,
            to_ascii=True,
        )
        return {"text": text}

    for domain in DOMAINS:
        for split in ["train", "valid"]:
            dataset = datasets.load_dataset("liaad/PtBrVId", domain, split=split)
            dataset = dataset.map(clean_text, num_proc=mp.cpu_count())
            dataset = dataset.filter(lambda x: x["text"] != "", num_proc=mp.cpu_count())
            dataset.push_to_hub("liaad/PtBrVId", domain, split=split)


def drop_outliers():
    def count_tokens(example):
        tokens = word_tokenize(example["text"], "portuguese")
        return {"n_tokens": len(tokens)}

    for domain in DOMAINS:
        trainset = datasets.load_dataset("liaad/PtBrVId", domain, split="train")
        validset = datasets.load_dataset("liaad/PtBrVId", domain, split="valid")
        dataset = datasets.concatenate_datasets([trainset, validset])
        dataset = dataset.map(count_tokens, num_proc=mp.cpu_count())
        q1 = np.percentile(dataset["n_tokens"], 25) 
        q3 = np.percentile(dataset["n_tokens"], 75) 
        iqr = q3 - q1
        min_tokens = q1 - 1.5 * iqr
        max_tokens = q3 + 1.5 * iqr
        dataset = dataset.filter(lambda x: min_tokens < x["n_tokens"] < max_tokens, num_proc=mp.cpu_count())
        dataset = dataset.remove_columns("n_tokens")

        label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
        valid_label0 = label0.select(range(N))
        train_label0 = label0.select(range(N, len(label0)))
        label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
        valid_label1 = label1.select(range(N))
        train_label1 = label1.select(range(N, len(label1)))

        trainset = datasets.concatenate_datasets([train_label0, train_label1])
        validset = datasets.concatenate_datasets([valid_label0, valid_label1])

        trainset.push_to_hub("liaad/PtBrVId", domain, split="train")
        validset.push_to_hub("liaad/PtBrVId", domain, split="valid")


def train_valid_split():
    for domain in DOMAINS:
        trainset = datasets.load_dataset("liaad/PtBrVId", domain, split="train")
        validset = datasets.load_dataset("liaad/PtBrVId", domain, split="valid")
        dataset = datasets.concatenate_datasets([trainset, validset])

        # add new valid examples
        label0 = dataset.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
        valid_label0 = label0.select(range(N))
        train_label0 = label0.select(range(N, len(label0)))
        label1 = dataset.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
        valid_label1 = label1.select(range(N))
        train_label1 = label1.select(range(N, len(label1)))

        trainset = datasets.concatenate_datasets([train_label0, train_label1])
        validset = datasets.concatenate_datasets([valid_label0, valid_label1])

        trainset.push_to_hub("liaad/PtBrVId", domain, split="train")
        validset.push_to_hub("liaad/PtBrVId", domain, split="valid")


if __name__ == "__main__":
    drop_outliers()
