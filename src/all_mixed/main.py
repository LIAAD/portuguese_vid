import torch
import traceback
from pathlib import Path
import os
import time
from data import pre_process_dataset
from model import LanguageIdentifer
from trainer import Trainer
from stats import plot_chart, print_to_file

CURRENT_PATH = Path(__file__).parent


def main():
    filename = time.strftime("%Y%m%d-%H%M%S")

    output_path = os.path.join(CURRENT_PATH, 'out', filename)

    os.makedirs(output_path, exist_ok=True)

    TRAIN_CONDITIONS = {
        'lr': 1e-5,
        'epochs': 50,
        'bertimbau_layers_to_freeze': 0,
        'pos_layers_to_freeze': 0,
        'early_stopping': 3,
        'bert_max_size': 512
    }

    type_of_embeddings = 'pos_only'

    final_output_path = os.path.join(output_path)

    os.makedirs(final_output_path, exist_ok=True)

    train_dataloader = pre_process_dataset(TRAIN_CONDITIONS['bert_max_size'])

    model = LanguageIdentifer(mode=type_of_embeddings, pos_layers_to_freeze=TRAIN_CONDITIONS[
                              'pos_layers_to_freeze'], bertimbau_layers_to_freeze=TRAIN_CONDITIONS['bertimbau_layers_to_freeze'])
    
    trainer = Trainer(model, train_dataloader, training_conditions = TRAIN_CONDITIONS, output_dir=final_output_path)

    results_df = trainer.train()

    plot_chart(results_df, final_output_path)

    print_to_file(results_df=results_df, test_accuracy=None, output_path=final_output_path, TRAIN_CONDITIONS=TRAIN_CONDITIONS, type_of_embeddings=type_of_embeddings)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()
