import logging

import torch

from ptvid.constants import DEVICE, MODEL_DIR


class Trainer:
    def __init__(self, train_key: str, model, train_loader, valid_loader, criterion, optimizer, scheduler=None):
        self.model_path = MODEL_DIR / train_key / "model.pth"
        self.model = model.to(DEVICE)
        self.device = DEVICE
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = 3
        self.early_stopping_counter = 0
        self.best_valid_loss = float("inf")

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
            probs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
            loss = self.criterion(probs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predicted = (probs > 0.5).to(torch.int)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                train_loss = running_loss / (batch_idx + 1)
                train_acc = 100.0 * correct / total
                logging.info(f"[{epoch}, {batch_idx}] loss: {train_loss:.4f}, accuracy: {train_acc:.2f}%")

        if self.scheduler:
            self.scheduler.step(train_loss)

        train_loss = running_loss / (batch_idx + 1)
        train_acc = 100.0 * correct / total
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

                probs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
                loss = self.criterion(probs, labels)

                running_loss += loss.item()
                predicted = (probs > 0.5).to(torch.int)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        valid_loss = running_loss / len(self.valid_loader)
        valid_acc = 100.0 * correct / total
        logging.info(f"Validation loss: {valid_loss:.4f}, accuracy: {valid_acc:.2f}%")

        return valid_loss, valid_acc

    def fit(self, epochs: int) -> dict:
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate()

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_model()

            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logging.info("Early stopping triggered!")
                    break

        best_state_dict = torch.load(self.model_path)
        self.model.load_state_dict(best_state_dict)

        metrics = {"train": {"loss": train_loss, "acc": train_acc}, "valid": {"loss": valid_loss, "acc": valid_acc}}
        return metrics

    def save_model(self):
        if not self.model_path.exists():
            self.model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), str(self.model_path))
