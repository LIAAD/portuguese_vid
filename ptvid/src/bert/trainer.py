import logging

import torch
import torch.nn as nn
import torch.optim as optim

from ptvid.constants import DEVICE


class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler=None):
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

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (batch) in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE, dtype=torch.float)

                outputs = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)
                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.valid_loader)
        val_accuracy = 100.0 * correct / total
        logging.info(f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.2f}%")

        return val_loss, val_accuracy

    def fit(self, epochs):
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


# Example usage
if __name__ == "__main__":
    # Define model, criterion, optimizer, scheduler, and dataloaders here
    model = ...  # Your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = ...  # Your training dataloader
    val_loader = ...  # Your validation dataloader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.fit(epochs=20)
    trainer.save_model("model.pth")
