import json
import logging
import multiprocessing as mp
from collections import Counter

import datasets
import nltk
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, pipeline

from ptvid.constants import DOMAINS, LABEL2ID
from ptvid.src.bert.model import LanguageIdentifier
from ptvid.src.bert.trainer import Trainer
from ptvid.src.delexicalizer import Delexicalizer

logging.basicConfig(level=logging.INFO)

DELEX = True
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 2
LR = 1e-5

train_dataset = []
for domain in DOMAINS:
    domain_data = datasets.load_dataset("u1537782/PtBrVId", domain, split="train")
    label0 = domain_data.filter(lambda x: x["label"] == 0, num_proc=mp.cpu_count())
    label1 = domain_data.filter(lambda x: x["label"] == 1, num_proc=mp.cpu_count())
    train_dataset.append(label0.select(range(3000)))
    train_dataset.append(label1.select(range(3000)))
train_dataset = datasets.concatenate_datasets(train_dataset)

valid_dataset = datasets.concatenate_datasets(
    [datasets.load_dataset("u1537782/PtBrVId", domain, split="valid") for domain in DOMAINS]
)


if DELEX:
    delexicalizer = Delexicalizer(prob_pos_tag=0.6, prob_ner_tag=0.2)
    train_dataset = train_dataset.map(
        lambda x: {"text": delexicalizer.delexicalize(x["text"])}, num_proc=mp.cpu_count()
    )

model = LanguageIdentifier(model_name=MODEL_NAME)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


def _adapt_dataset(dataset):
    def _tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

    dataset = dataset.map(_tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DataLoader(dataset, batch_size=BATCH_SIZE)


train_loader = _adapt_dataset(train_dataset)
valid_loader = _adapt_dataset(valid_dataset)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, verbose=True)

trainer = Trainer(
    train_key="test",
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
)

metrics = trainer.fit(epochs=EPOCHS)


## EVALUATION
def load_dsl():
    dsl = datasets.load_dataset("LCA-PORVID/dsl_tl", split="test")
    dsl = dsl.filter(lambda x: x["label"] in [0, 1])
    return dsl


def load_frmt():
    frmt = datasets.load_dataset("u1537782/frmt", split="test")
    pt = [text for text in frmt["pt"] if text]
    br = [text for text in frmt["br"] if text]

    labels = [0] * len(pt) + [1] * len(br)
    text = pt + br

    return datasets.Dataset.from_dict({"text": text, "label": labels})


# fmrt
print("FRMT")
frmt = load_frmt()
pred = model.predict(frmt["text"])
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
dsl = load_dsl()

pred = model.predict(dsl["text"])
metrics = {
    "accuracy": accuracy_score(dsl["label"], pred),
    "precision": precision_score(dsl["label"], pred),
    "recall": recall_score(dsl["label"], pred),
    "f1": f1_score(dsl["label"], pred),
}
print(metrics)
print(classification_report(dsl["label"], pred, target_names=["PT", "BR"], digits=4))
