import json
from collections import Counter

import datasets
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import pipeline

from ptvid.constants import LABEL2ID

MODELS = [
    "liaad/LVI_bert-base-portuguese-cased",
    "liaad/LVI_bert-large-portuguese-cased",
    "liaad/LVI_albertina-100m-portuguese-ptpt-encoder",
    "liaad/LVI_albertina-900m-portuguese-ptpt-encoder",
]


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


def evaluate(pipe, dataset):
    pred = pipe(dataset["text"])
    pred = [LABEL2ID[p["label"]] for p in pred]
    metrics = {
        "accuracy": accuracy_score(dataset["label"], pred),
        "precision": precision_score(dataset["label"], pred),
        "recall": recall_score(dataset["label"], pred),
        "f1": f1_score(dataset["label"], pred),
    }

    print(classification_report(dataset["label"], pred, target_names=["PT", "BR"]))
    return metrics


def main():
    frmt = load_frmt()
    dsl = load_dsl()

    results = {"frmt": {}, "dsl": {}}
    for model_name in MODELS:
        print(f"Model: {model_name}")
        print("FRMT")
        pipe = pipeline("text-classification", model=model_name, device="cuda")
        results["frmt"][model_name] = evaluate(pipe, frmt)

        print("DSL")
        print(f"Model: {model_name}")
        pipe = pipeline("text-classification", model=model_name, device="cuda")
        results["dsl"][model_name] = evaluate(pipe, dsl)
    json.dump(results, open("evaluation.json", "w"), indent=4)


if __name__ == "__main__":
    main()
