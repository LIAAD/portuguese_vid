import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from tester import Tester
import os


class Trainer():
    def __init__(self, model, train_dataloader, training_conditions, output_dir):
        self.model = model
        self.train_dataloader = train_dataloader
        self.training_conditions = training_conditions
        self.output_dir = output_dir

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.training_conditions['lr'])

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(
            train_dataloader.dataset['label']), y=train_dataloader.dataset['label'].numpy()).tolist()

        print(f"Class Weights: {self.class_weights}")

        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.class_weights[1]).to(self.device))

        self.tester = Tester(model, self.loss_fn)

        self.model.to(self.device)

        self.results = pd.DataFrame(
            columns=['epoch', 'train_loss', 'validation_loss', 'validation_accuracy', 'validation_f1', 'validation_precision', 'validation_recall'])

    def epoch_iter(self):
        total_loss = 0

        self.model.train()

        with torch.enable_grad():
            for batch in tqdm(self.train_dataloader):

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Convert Labels from 1D to 2D. Example [4] -> [4x1]
                labels = batch['label'].unsqueeze(1).float().to(self.device)

                pred = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask)

                loss_fn = self.loss_fn(pred, labels)

                self.optimizer.zero_grad()

                loss_fn.backward()

                self.optimizer.step()

                total_loss += loss_fn.item()

        return total_loss / len(self.train_dataloader)

    def train(self):
        best_accuracy = -math.inf
        best_validation_loss = math.inf
        current_tries = self.training_conditions['early_stopping']

        epochs_done = 0

        for epoch in tqdm(range(self.training_conditions['epochs']), miniters=10, ascii=True):

            if current_tries == 0:
                print('Early stopping...')
                break

            train_loss = self.epoch_iter()

            validation_accuracy, validation_loss, f1, precision, recall = self.tester.test()

            print(
                f"Epoch: {epoch} | Train Loss: {train_loss} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_accuracy}, Validation F1: {f1}, Validation Precision: {precision}, Validation Recall: {recall}")

            if validation_accuracy >= best_accuracy and validation_loss <= best_validation_loss:
                best_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                print('\n')
                print('Saving model...\n')
                print('\n')
                torch.save(self.model.state_dict(), os.path.join(
                    self.output_dir, 'best_accuracy.pt'))

                current_tries = self.training_conditions['early_stopping']

            else:
                current_tries -= 1
                print(f"Early stopping tries left: {current_tries}")

            # Append results to dataframe
            self.results = pd.concat([self.results, pd.DataFrame({'epoch': [epoch], 'train_loss': [train_loss], 'validation_loss': [validation_loss], 'validation_accuracy': [
                                     validation_accuracy], 'validation_f1': [f1], 'validation_precision': [precision], 'validation_recall': [recall]})])

            if best_validation_loss <= 0.1 and epochs_done > 0:
                print("Validation Loss is Very Low. Can Stop")
                break

            epochs_done += 1

        self.training_conditions['epochs_done'] = epochs_done
        self.training_conditions['class_weights'] = self.class_weights
        self.training_conditions['number_docs'] = len(
            self.train_dataloader.dataset)

        return self.results
