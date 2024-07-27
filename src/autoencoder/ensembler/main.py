import torch
import traceback
import pandas as pd
from model import EnsembleModel
from datasets import load_dataset
import evaluate
import os
from data import tokenize, create_dataloader
from pathlib import Path
from tqdm import tqdm
import numpy as np

CURRENT_PATH = Path(__file__).parent

def process_results(results):
    predictions = []

    # Perform Majority Voting
    for row in results:
        number_of_ones = np.array(row).sum()
        
        number_of_zeros = len(row) - number_of_ones
        
        if number_of_ones > number_of_zeros:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
        



def test(dataset, ensemble):
    with torch.no_grad():
        ensemble.eval()
              
        all_labels = []
        
        predictions = []
                
        for batch in tqdm(dataset, ascii=True, miniters=10):
            input_ids = batch['input_ids'].to(ensemble.device)
            attention_mask = batch['attention_mask'].to(ensemble.device)
            labels = batch['label'].to(ensemble.device)

            results = ensemble(input_ids, attention_mask)
                    
            all_labels.extend(labels.flatten().int().cpu().tolist())
            predictions.extend(process_results(results))

        return predictions, all_labels
            
def cycle_dataset(dataset, domain, df_results, ensemble):
    dataset = tokenize(dataset)
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])

    dataset = create_dataloader(dataset)

    evaluate_accuracy = evaluate.load('accuracy')
    evaluate_f1 = evaluate.load('f1')
    evaluate_precision = evaluate.load('precision')
    evaluate_recall = evaluate.load('recall')

    predictions, all_labels = test(dataset, ensemble)

    accuracy = evaluate_accuracy.compute(
        predictions=predictions, references=all_labels)['accuracy']
    precision = evaluate_precision.compute(
        predictions=predictions, references=all_labels)['precision']
    recall = evaluate_recall.compute(
        predictions=predictions, references=all_labels)['recall']
    f1 = evaluate_f1.compute(
        predictions=predictions, references=all_labels)['f1']
    
    df_results = pd.concat([df_results, pd.DataFrame({
        'domain': domain,            
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }, index=[0])])

    df_results.to_json(os.path.join(CURRENT_PATH, "out", "results.json"), orient="records")

    return df_results


def main():
    df_results = pd.DataFrame(columns=['domain', 'accuracy', 'precision', 'recall', 'f1'])
    
    domains = ['politics', 'news', 'law', 'social_media', 'literature', 'web']
    
    ensemble = EnsembleModel(domains=domains)

    for domain in ['dslcc']:
        dataset = load_dataset("arubenruben/portuguese_dslcc", domain, split='test')
        df_results = cycle_dataset(
            dataset, domain, df_results, ensemble)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()