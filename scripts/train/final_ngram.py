import json
import logging
import multiprocessing as mp
from collections import Counter

import datasets
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from transformers import pipeline

from ptvid.src.delexicalizer import Delexicalizer
from ptvid.constants import DOMAINS, LABEL2ID

DELEX = True

PARAMS = {
    "tfidf__max_features": [100, 500, 1000, 5000, 10000, 50000, 100000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 10)],
    "tfidf__lowercase": [True, False],
    "tfidf__analyzer": ["word", "char"],
}


train_dataset = []
for domain in DOMAINS:
    domain_data = datasets.load_dataset("liaad/PtBrVId", domain, split="train")
    label0 = domain_data.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
    label1 = domain_data.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
    train_dataset.append(label0.select(range(3000)))
    train_dataset.append(label1.select(range(3000)))
train_dataset = datasets.concatenate_datasets(train_dataset)


if DELEX:
    delexicalizer = Delexicalizer(prob_pos_tag=0.2, prob_ner_tag=0.6)
    train_dataset = train_dataset.map(
        lambda x: {"text": delexicalizer.delexicalize(x["text"])}, num_proc=mp.cpu_count()
    )


pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                tokenizer=lambda text: word_tokenize(text, language="portuguese"),
                stop_words=nltk.corpus.stopwords.words("portuguese"),
                token_pattern=None,
            ),
        ),
        ("clf", BernoulliNB()),
    ]
)


search = RandomizedSearchCV(
    pipeline,
    PARAMS,
    scoring="f1_macro",
    n_jobs=-1,
    n_iter=500,
    cv=StratifiedKFold(n_splits=2, random_state=42, shuffle=True),
    error_score="raise",
)

results = search.fit(np.array(train_dataset["text"]), np.array(train_dataset["label"]))
results = results.best_estimator_


def load_dsl():
    dsl = datasets.load_dataset("LCA-PORVID/dsl_tl", split="test")
    dsl = dsl.filter(lambda x: x["label"] in [0, 1])
    return dsl


def load_frmt():
    frmt = datasets.load_dataset("hugosousa/frmt", split="test")
    pt = [text for text in frmt["pt"] if text]
    br = [text for text in frmt["br"] if text]

    labels = [0] * len(pt) + [1] * len(br)
    text = pt + br

    return datasets.Dataset.from_dict({"text": text, "label": labels})


frmt = load_frmt()
dsl = load_dsl()

frmt_pred = results.predict(frmt["text"])
dsl_pred = results.predict(dsl["text"])

# fmrt
print("FRMT")
pred = results.predict(frmt["text"])
metrics = {
    "accuracy": accuracy_score(frmt["label"], pred),
    "precision": precision_score(frmt["label"], pred),
    "recall": recall_score(frmt["label"], pred),
    "f1": f1_score(frmt["label"], pred),
}
print(metrics)
print(classification_report(frmt["label"], pred, target_names=["PT", "BR"], digits=4))


# dsl
print("DSL")
pred = results.predict(dsl["text"])
metrics = {
    "accuracy": accuracy_score(dsl["label"], pred),
    "precision": precision_score(dsl["label"], pred),
    "recall": recall_score(dsl["label"], pred),
    "f1": f1_score(dsl["label"], pred),
}
print(metrics)
print(classification_report(dsl["label"], pred, target_names=["PT", "BR"], digits=4))
