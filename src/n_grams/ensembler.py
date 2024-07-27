import os
from pathlib import Path
import pickle
from datasets import load_dataset
import evaluate
import numpy as np
import nltk
import json
from tqdm import tqdm

CURRENT_PATH = Path(__file__).parent

def process_strategy(strategy, raw_predictions):
    raw_predictions = np.array(raw_predictions)

    if strategy == 'mean':
        predictions = np.mean(raw_predictions, axis=0)
    
    elif strategy == 'median':
        predictions = np.median(raw_predictions, axis=0)
    
    elif strategy == 'majority_voting':
        raw_predictions = raw_predictions.transpose()

        predictions = [1 if np.sum(row > 0) > np.sum(row < 0) else -1 for row in raw_predictions]
    
    else:
        raise Exception(f"Strategy {strategy} not implemented")
    
    predictions = np.array(predictions)

    # Convert predictions to 1 or 0 based on a threshold
    return np.where(predictions > 0.5, 1, 0)

def tokenizer(text):
    return nltk.tokenize.word_tokenize(text, language="portuguese")

def process_batch(batch, models, strategy):
    predictions = []

    for model in models:
        predictions.append(model.predict(batch).tolist())                                                                        
    
    return process_strategy(strategy, predictions)
        

def main():
    models = []

    accuracy_evaluate = evaluate.load('accuracy')
    f1_evaluate = evaluate.load('f1')
    precision_evaluate = evaluate.load('precision')
    recall_evaluate = evaluate.load('recall')

    results = {}

    for domain in ['politics', 'news', 'law', 'social_media', 'literature', 'web']:
        with open(os.path.join(CURRENT_PATH, "out", "models", f"{domain}.pickle"), "rb") as f:
            models.append(pickle.load(f))

    print(f"Loaded {len(models)} models")

    for domain in ['politics', 'news', 'law', 'social_media', 'literature', 'web']:
        
        dataset = load_dataset('arubenruben/Portuguese_Language_Identification', domain, split="test")        
        
        results[domain]={
            'max':{},
            'mean': {},
            'median': {},
            'majority_voting':{}
        }
        
        
        for strategy in ['majority_voting', 'mean', 'median']:
            
            batch = []
            predictions = []

            print(f"Processing {domain} with {strategy}")
            
            for row in tqdm(dataset):
                
                batch.append(row['text'])

                if len(batch) == 100:                    
                    predictions.extend(process_batch(batch, models, strategy))
                    batch = []

            if len(batch) > 0:                
                predictions.extend(process_batch(batch, models, strategy))                

            results[domain][strategy]['accuracy'] = accuracy_evaluate.compute(predictions=predictions, references = dataset['label'])['accuracy']
            results[domain][strategy]['f1'] = f1_evaluate.compute(predictions=predictions, references = dataset['label'])['f1']
            results[domain][strategy]['precision'] = precision_evaluate.compute(predictions=predictions, references = dataset['label'])['precision']
            results[domain][strategy]['recall'] = recall_evaluate.compute(predictions=predictions, references = dataset['label'])['recall']

            print(results[domain][strategy])

    json.dump(results, open(os.path.join(CURRENT_PATH, "out", "results_ensembler.json"), "w"))

    print(f"Loaded {len(models)} models")

if __name__ == "__main__":
    main()