import logging

import torch
import torch.nn as nn
import torch.optim as optim

from ptvid.constants import DEVICE, MODEL_DIR


class Trainer:
    def __init__(self, train_key: str, model, train_loader, valid_loader, criterion, optimizer, scheduler=None):
        self.model_path = MODEL_DIR / self._train_key / "model.pth"
        self.model = model
        self.device = DEVICE
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (batch) in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE, dtype=torch.float)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                logging.info(
                    f"[{epoch}, {batch_idx}] loss: {running_loss / (batch_idx + 1):.4f}, accuracy: {100. * correct / total:.2f}%"
                )

        if self.scheduler:
            self.scheduler.step()
        
        train_loss = running_loss / (batch_idx + 1)
        train_acc = 100. * correct / total
        return train_loss, train_acc 

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (batch) in enumerate(self.valid_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE, dtype=torch.float)

                outputs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_loss = running_loss / len(self.valid_loader)
        valid_acc = 100.0 * correct / total
        logging.info(f"Validation loss: {valid_loss:.4f}, accuracy: {valid_acc:.2f}%")

        return valid_loss, valid_acc

    def fit(self, epochs: int) -> dict:
        best_valid_loss = 10**5
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate()
            if valid_loss < best_valid_loss:
                self.save_model()

        best_state_dict = torch.load(self.model_path)
        self.model.load_state_dict(best_state_dict)

        metrics = {
            "train": {
                "loss": train_loss,
                "acc": train_acc
            },
            "valid": {
                "loss": valid_loss,
                "acc": valid_acc
            }
        }
        return metrics 

    def save_model(self, path):
        if not self.model_path.exists():
            self.model_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), str(self.model_path))
