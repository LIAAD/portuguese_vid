from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import os
import dotenv
import evaluate
import json
import ssl
import pickle
from imblearn.under_sampling import RandomUnderSampler
from datasets import Dataset
import pandas as pd

dotenv.load_dotenv(dotenv.find_dotenv())

SELECT_MAX_VALUE = 40000


class DensifyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


CURRENT_PATH = Path(__file__).parent


def tokenizer(text):
    return nltk.tokenize.word_tokenize(text, language="portuguese")


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download("stopwords")
nltk.download("punkt")

stop_words = nltk.corpus.stopwords.words("portuguese")

PARAM_GRID = [
    {
        'tokenizer': tokenizer,
        'max_features': 40000,
        'max_df': 1.0,
        'lowercase': False,
        'stop_words': stop_words,
        'token_pattern': None,
        'ngram_range': (1, 2),
        'analyzer': "word"
    },
    {
        'tokenizer': tokenizer,
        'max_features': 40000,
        'max_df': 1.0,
        'lowercase': False,
        'stop_words': stop_words,
        'token_pattern': None,
        'ngram_range': (1, 5),
        'analyzer': "char_wb"
    }
]

def balance_data(dataset):

    df = dataset.to_pandas()

    print(f"Before balancing: {df['label'].value_counts()}")

    rus = RandomUnderSampler(random_state=42, replacement=True)

    X_resampled, y_resampled = rus.fit_resample(
        df['text'].to_numpy().reshape(-1, 1), df['label'].to_numpy())

    df = pd.DataFrame(
        {'text': X_resampled.flatten(), 'label': y_resampled})
        
    print(f"After balancing: {df['label'].value_counts()}")

    return Dataset.from_pandas(df)


def train(domain, params):
    
    train_dataset = load_dataset('arubenruben/Portuguese_Language_Identification', domain, split="train")
    
    train_dataset = balance_data(train_dataset)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=params['tokenizer'],
                                  max_features=params['max_features'],
                                  max_df=params['max_df'],
                                  lowercase=params['lowercase'],
                                  stop_words=params['stop_words'],
                                  token_pattern=params['token_pattern'],
                                  ngram_range=params['ngram_range'],
                                  analyzer=params['analyzer'])),        
        ("clf", MultinomialNB()),
    ])

    pipeline.fit(train_dataset["text"], train_dataset["label"])

    return pipeline


def benchmark(pipeline, params, train_domain):
    
    output_metrics = {
        'max_features': params['max_features'],
        'max_df': params['max_df'],
        'lowercase': params['lowercase'],
        'n_gram_range': params['ngram_range'],
        'analyzer': params['analyzer'],
        'train_domain' : train_domain,
        'results':[]
    }

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
            
    for domain in ['politics', 'news', 'law', 'social_media', 'literature', 'web']:
            
        dataset = load_dataset('arubenruben/Portuguese_Language_Identification', domain, split="test")

        data
        
        print(f"Testing {domain} domain with {domain} dataset")
        
        predictions = pipeline.predict(dataset["text"])
        
        predictions = predictions.tolist()
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=dataset["label"])['accuracy']
        
        f1 = f1_metric.compute(predictions=predictions, references=dataset["label"])['f1']
        
        precision = precision_metric.compute(predictions=predictions, references=dataset["label"])['precision']
        
        recall = recall_metric.compute(predictions=predictions, references=dataset["label"])['recall']
                    
        output_metrics['results'].append({            
            'accuracy': accuracy,
            'test_domain': domain,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        })    

    with open(os.path.join(CURRENT_PATH, "out", "output_metrics.json"), "a") as f:
        json.dump(output_metrics, f)


def main():
    # Clear File Content
    with open(os.path.join(CURRENT_PATH, "out", "output_metrics.json"), "w") as f:
        f.close()
    
    for domain in ['politics', 'news', 'law', 'social_media', 'literature', 'web']:
        for params in PARAM_GRID:        
            print(f"Training {domain} domain")                        

            pipeline = train(domain, params)
            
            # Save Model
            with open(os.path.join(CURRENT_PATH, "out", "models", f"{domain}.pickle"), "wb") as f:                
                pickle.dump(pipeline, f)

            print(f"Trained {domain} domain")
            
            benchmark(pipeline, params, domain)


if __name__ == "__main__":
    main()
