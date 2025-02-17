import json
import logging
import os
from time import time

from fire import Fire
import numpy as np
import torch

from ptvid.constants import DATASET_NAME, RESULTS_DIR, SAMPLE_SIZE
from ptvid.src.bert.data import Data
from ptvid.src.bert.model import LanguageIdentifier
from ptvid.src.bert.trainer import Trainer
from ptvid.src.utils import create_output_dir, setup_logger


def main(
    dataset_name: str = DATASET_NAME,
    domain: str = "literature",
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    batch_size: int = 64,
    epochs: int = 30,
    patience: int = 3,
    lr: float = 1e-5,
):
    current_path = os.path.dirname(os.path.abspath(__file__))
    current_time = int(time())
    create_output_dir(current_path, current_time)
    setup_logger(current_path, current_time)

    train_data = Data(dataset_name, split="train", tokenizer_name=model_name, batch_size=batch_size)
    valid_data = Data(dataset_name, split="valid", tokenizer_name=model_name, batch_size=batch_size)

    for pos_prob in np.arange(0.0, 1.1, 0.1):
        for ner_prob in np.arange(0.0, 1.1, 0.1):
            train_key = f"{domain}_pos_prob={pos_prob}_ner_prob={ner_prob}"
            outpath = RESULTS_DIR / train_key / "metrics.json"

            if not outpath.exists():
                outpath.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Running {domain} pos_prob={pos_prob}, ner_prob={ner_prob}")
                train_loader = train_data.load_domain(
                    domain, balance=True, pos_prob=pos_prob, ner_prob=ner_prob, sample_size=SAMPLE_SIZE
                )
                valid_loader = valid_data.load_domain(domain, balance=True)

                model = LanguageIdentifier(model_name)

                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

                trainer = Trainer(
                    train_key=train_key,
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                metrics = trainer.fit(epochs=epochs)
                json.dump(metrics, outpath.open("w"), indent=4)

            else:
                logging.info(f"This model has already been trained with this configuration. Check {outpath}")


if __name__ == "__main__":
    Fire(main)
