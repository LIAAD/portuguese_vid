import logging

import torch
from pt_variety_identifier.src.bert.model import LanguageIdentfier
from tqdm import tqdm


class Trainer:
    def __init__(self, train_dataset, params) -> None:
        self.train_dataset = train_dataset
        print(f"Using {self.device} device")

        self.model = LanguageIdentfier(params["model_name"])

        self.epochs = params["epochs"]
        self.lr = 1e-5
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.early_stoping = params["early_stoping"]
        self.device = params["device"]

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.early_stoping // 2, verbose=True
        )

    def _epoch_iter(self):
        self.model.train()
        self.model.to(self.device)

        with torch.enable_grad():
            total_loss = 0

            for batch in tqdm(self.train_dataset):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device, dtype=torch.float)

                outputs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            self.scheduler.step(total_loss)

            return total_loss / len(self.train_dataset)

    def train(self):
        logging.info(f"Training model in {self.device}...")

        for epoch in tqdm(range(self.epochs)):
            loss = self._epoch_iter()

            logging.info(f"Epoch {epoch} Loss: {loss}")

            if loss < 0.1:
                logging.info(f"Loss is too low, stoping training...")
                break

        # TODO: Return Training metrics
        return [], self.model
