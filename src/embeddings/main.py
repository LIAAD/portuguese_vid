from model import LanguageIdentifer
from data import pre_process_dataset, load_test_set
import dotenv
from trainer import Trainer
from tester import Tester
from huggingface_hub import login
import os
from pathlib import Path
import time
from stats import print_to_file, plot_chart
import torch
import traceback
import pandas as pd

CURRENT_PATH = Path(__file__).parent

dotenv.load_dotenv(dotenv.find_dotenv())

login(token=os.getenv("HF_TOKEN"))


def benchmark(model, train_domain):
    df_results = pd.DataFrame(columns=[
                              'test_domain', 'test_accuracy', 'test_loss', 'test_f1', 'test_precision', 'test_recall'])

    for domain in ['literature', 'web', 'politics', 'social_media', 'law', 'news']:
        test_dataloader = load_test_set(domain)

        tester = Tester(model, test_dataloader)

        accuracy, loss, f1, precision, recall = tester.test()

        df_results = pd.concat([df_results, pd.DataFrame([[domain, accuracy, loss, f1, precision, recall]], columns=[
                               'test_domain', 'test_accuracy', 'test_loss', 'test_f1', 'test_precision', 'test_recall'])])

    df_results.to_json(CURRENT_PATH, "out", )

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

    for domain in ['dslcc']:

        type_of_embeddings = 'pos_only'

        final_output_path = os.path.join(output_path, domain)

        os.makedirs(final_output_path, exist_ok=True)

        train_dataloader, test_dataloader = pre_process_dataset(
            domain, TRAIN_CONDITIONS['bert_max_size'])

        model = LanguageIdentifer(mode=type_of_embeddings, pos_layers_to_freeze=TRAIN_CONDITIONS[
                                  'pos_layers_to_freeze'], bertimbau_layers_to_freeze=TRAIN_CONDITIONS['bertimbau_layers_to_freeze'])

        trainer = Trainer(model, train_dataloader,
                          test_dataloader, training_conditions=TRAIN_CONDITIONS, output_dir=final_output_path )

        results_df = trainer.train()

        plot_chart(results_df, final_output_path)

        print_to_file(results_df=results_df, test_accuracy=None, output_path=final_output_path,
                      TRAIN_CONDITIONS=TRAIN_CONDITIONS, type_of_embeddings=type_of_embeddings)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()
