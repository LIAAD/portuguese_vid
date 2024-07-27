import logging

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self, train_dataset, params, n_iter=500) -> None:
        nltk.download("stopwords")
        nltk.download("punkt")

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=lambda text: word_tokenize(text, language="portuguese"),
                        stop_words=nltk.corpus.stopwords.words("portuguese"),
                    ),
                ),
                ("clf", BernoulliNB()),
            ]
        )

        self.params = params
        self.n_iter = n_iter

        self.cv = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

        self.search = RandomizedSearchCV(
            self.pipeline,
            self.params,
            scoring="f1_macro",
            n_jobs=-1,
            n_iter=self.n_iter,
            cv=self.cv,
            error_score="raise",
        )

        self.train_dataset = train_dataset

    def train(self):
        logging.info("Training model...")

        results = self.search.fit(np.array(self.train_dataset["text"]), np.array(self.train_dataset["label"]))

        logging.info("Training finished!")

        return results, results.best_estimator_
