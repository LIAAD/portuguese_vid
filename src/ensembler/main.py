import torch
import dotenv
import os
from datasets import load_dataset
from data import tokenize, create_dataloader
from model import EnsembleModel
from tqdm import tqdm
from huggingface_hub import login
import torch
import evaluate
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

dotenv.load_dotenv(dotenv.find_dotenv())

login(token=os.getenv("HF_TOKEN"))

CURRENT_PATH = Path(__file__).parent


def process_output(predictions, reduction):
    final_predictions = []

    for tensor in predictions:
        if reduction == 'mean':
            raw_label = torch.mean(tensor).item()
        elif reduction == 'median':
            raw_label = torch.median(tensor).item()
        elif reduction == 'max':
            max_value = torch.max(tensor)
            min_value = torch.min(tensor)

            raw_label = min_value.item() if abs(min_value) > max_value else max_value.item()
        elif reduction == 'majority_vote':
            number_of_positive = torch.sum(tensor > 0).item()
            number_of_negative = len(tensor) - number_of_positive

            raw_label = 1 if number_of_positive > number_of_negative else -1
        else:
            raise ValueError("Invalid reduction type")

        final_predictions.append(raw_label)

    final_predictions = torch.tensor(
        final_predictions, dtype=torch.float32, device=predictions.device)

    return (final_predictions > 0).flatten().int().cpu().tolist()


def test(dataset, ensemble):

    with torch.no_grad():
        ensemble.eval()

        predictions_acum = {
            'mean': [],
            'median': [],
            'max': [],
            'majority_vote': [],
        }

        all_labels = []

        for batch in tqdm(dataset):
            input_ids = batch['input_ids'].to(ensemble.device)
            attention_mask = batch['attention_mask'].to(ensemble.device)

            # Convert Labels from 1D to 2D. Example [4] -> [4x1]
            labels = batch['label'].unsqueeze(1).float().to(ensemble.device)

            all_labels.extend(labels.flatten().int().cpu().tolist())

            outputs = ensemble(input_ids=input_ids,
                               attention_mask=attention_mask)

            for reduction in ['mean', 'median', 'max', 'majority_vote']:
                predictions_acum[reduction].extend(
                    process_output(outputs, reduction))

        return predictions_acum, all_labels


def cycle_dataset(dataset, domain, df_results, ensemble):
    dataset = tokenize(dataset)
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'attention_mask', 'label'])

    dataset = create_dataloader(dataset)

    evaluate_accuracy = evaluate.load('accuracy')
    evaluate_f1 = evaluate.load('f1')
    evaluate_precision = evaluate.load('precision')
    evaluate_recall = evaluate.load('recall')

    predictions_acum, all_labels = test(dataset, ensemble)

    for reduction in ['mean', 'median', 'max', 'majority_vote']:
        accuracy = evaluate_accuracy.compute(
            predictions=predictions_acum[reduction], references=all_labels)['accuracy']
        precision = evaluate_precision.compute(
            predictions=predictions_acum[reduction], references=all_labels)['precision']
        recall = evaluate_recall.compute(
            predictions=predictions_acum[reduction], references=all_labels)['recall']
        f1 = evaluate_f1.compute(
            predictions=predictions_acum[reduction], references=all_labels)['f1']

        df_results = pd.concat([df_results, pd.DataFrame({
            'domain': domain,
            'reduction': reduction,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }, index=[0])])

    df_results.to_json(os.path.join(CURRENT_PATH, "out", "results.json"), orient="records")

    return df_results


def main():
    df_results = pd.DataFrame(
        columns=['domain', 'reduction', 'accuracy', 'precision', 'recall', 'f1'])

    ensemble = EnsembleModel()
    """
    for domain in ['d']:
        dataset = load_dataset(
            "arubenruben/Portuguese_Language_Identification", domain, split='test')
        
        dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), 50_000)))
        
        df_results = cycle_dataset(
            dataset, domain, df_results, ensemble)
    """

    dataset = load_dataset("arubenruben/portuguese_dslcc", split='test')
    df_results = cycle_dataset(dataset, 'dslcc', df_results, ensemble)
    
    torch.save(ensemble.state_dict(), os.path.join(
        CURRENT_PATH, "out", 'ensemble.pt'))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()
