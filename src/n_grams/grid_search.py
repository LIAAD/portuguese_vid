from ftlangdetect import detect
from datasets import DatasetDict, load_dataset, concatenate_datasets, Dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import evaluate
from pathlib import Path
import pickle
import json
import logging
import pandas as pd
import nltk
import dotenv
import os
import time
import ssl

CURRENT_PATH = Path(__file__).parent

THRESHOLD = 0.7

logger = logging.getLogger(__name__)

filename = None

dotenv.load_dotenv(dotenv.find_dotenv())

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')


def load_data():
    train_datasets = []
    test_datasets = []

    for dataset in json.loads(os.getenv("TRAIN_DATASETS")):
        dataset = load_dataset(dataset)

        train_datasets.append(dataset['train'])
        train_datasets.append(dataset['validation'])

    for dataset in json.loads(os.getenv("TEST_DATASETS")):
        test_datasets.append(dataset)

    return DatasetDict({
        "train": concatenate_datasets([dataset for dataset in train_datasets]).shuffle(),
    }), test_datasets


def fast_text_lang_detect(dataset, threshold=0.7):

    #with open("outliers.txt", "w", encoding='utf-8') as f:
    for split in ["train"]:

        pd_dataset = dataset[split].to_pandas()

        indexs_to_remove = []

        for index, row in pd_dataset.iterrows():

            result = detect(row["text"])

            if result['lang'] != "pt" or result["score"] < threshold:
                indexs_to_remove.append(index)
                #f.write(f"{row['text']} \t {result['lang']} \t {result['score']} \n")

        pd_dataset = pd_dataset.drop(index=indexs_to_remove)

        logger.debug(f"Removed {len(indexs_to_remove)} outliers")

        dataset[split] = Dataset.from_pandas(
            pd_dataset).select_columns(['text', 'label'])

    return dataset


def tokenizer(text):
    return nltk.tokenize.word_tokenize(text, language='portuguese')


def benchmark(grid_search, test_datasets_names):
    results = {}
    accuracy_metric = evaluate.load("accuracy")

    for dataset_name in test_datasets_names:
        try:
            print(f"Testing on {dataset_name}")

            dataset = load_dataset(dataset_name, split='test')

            y_true = dataset['label']
            y_pred = grid_search.predict(dataset['text'])

            y_pred = y_pred.tolist()

            accuracy = accuracy_metric.compute(
                references=y_true, predictions=y_pred)

            results[dataset_name] = accuracy['accuracy']

        except Exception as e:
            print(e)
            results[dataset_name] = "Error"
            print(f"Error but continuing")

    return results


def save_stuff(grid_search):
    global filename

    filename = time.strftime("%Y%m%d-%H%M%S")

    output_path = os.path.join(CURRENT_PATH.parent, 'out', filename)

    os.makedirs(output_path, exist_ok=True)

    # Save Best Model
    pickle.dump(grid_search.best_estimator_,
                open(os.path.join(output_path, "model.pkl"), "wb+"))

    pd.DataFrame(grid_search.cv_results_).to_json(
        os.path.join(output_path, "cv_results.json"), orient="records")

    grid_search.best_params_[
        "TRAIN_DATASETS"] = json.loads(os.getenv("TRAIN_DATASETS"))
    grid_search.best_params_[
        "TEST_DATASETS"] = json.loads(os.getenv("TEST_DATASETS"))

    grid_search.best_params_["train_accuracy"] = grid_search.best_score_

    json.dump(grid_search.best_params_, open(
        os.path.join(output_path, "best_params.json"), "w+"))


def save_benchmark_results(test_accuracy):
    global filename

    output_path = os.path.join(CURRENT_PATH.parent, 'out', filename)

    print(f"Saving benchmark results to {output_path}")

    with open(os.path.join(output_path, "best_params.json"), "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(os.path.join(output_path, "best_params.json"), "w", encoding="utf-8") as f:
        results["test_accuracy"] = test_accuracy
        json.dump(results, f)


class DensifyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


PARAM_GRID = {
    'tfidf__max_features': [40000, 50000, None],
    'tfidf__ngram_range': [(1, 3), (1, 4), (1, 5)],
    'tfidf__analyzer': ['char_wb'],
}

if __name__ == "__main__":
    dataset, test_datasets_names = load_data()

    stopwords = nltk.corpus.stopwords.words('portuguese')

    dataset = fast_text_lang_detect(dataset, threshold=THRESHOLD)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenizer,
         max_df=1.0, lowercase=False, stop_words=stopwords, token_pattern=None)),
        ('densify', DensifyTransformer()),
        ('clf', MultinomialNB()),
    ])

    grid_search = GridSearchCV(pipeline, PARAM_GRID, cv=2, verbose=11,
                               scoring='accuracy', return_train_score=True, refit='accuracy', n_jobs=-1)

    grid_search.fit(dataset["train"]["text"], dataset["train"]["label"])

    print("End of Grid Search")

    save_stuff(grid_search)

    test_accuracy = benchmark(grid_search, test_datasets_names)

    save_benchmark_results(test_accuracy)

    print("Completed")
