import torch
from model import AutoEncoder
from tqdm import tqdm
import pandas as pd
from tester import Tester
import math
from pathlib import Path
import os
import threading

CURRENT_PATH = Path(__file__).parent


class Trainer():
    def __init__(self, brazilian_train, european_train, domain, lr: float = 1e-3, n_epochs: int = 10):

        self.brazilian_model = AutoEncoder()
        self.european_model = AutoEncoder()
        self.domain = domain

        self.reconstruction_loss = torch.nn.MSELoss()

        self.optimizer_brazilian = torch.optim.Adam(
            self.brazilian_model.parameters(), lr=lr)
        self.optimizer_european = torch.optim.Adam(
            self.european_model.parameters(), lr=lr)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs

        self.brazilian_train = brazilian_train
        self.european_train = european_train

        self.tester = Tester(domain=domain)

    def epoch_iter(self, model, optimizer, train_dataloader):
        total_loss = 0

        for batch in tqdm(train_dataloader, miniters=10, ascii=True):
            input_ids = batch['input_ids'].to(self.device)

            attention_mask = batch['attention_mask'].to(self.device)

            bert_output, reconstruction = model(input_ids, attention_mask)

            train_loss = self.reconstruction_loss(reconstruction, bert_output)

            total_loss += train_loss.item()

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        return total_loss / len(train_dataloader)

    def train_model(self, model, optimizer, train_data):
        # Your existing code for training a single model here
        return_loss = self.epoch_iter(model, optimizer, train_data)
        

    def train(self):
        
        df_results = pd.DataFrame(columns=[
                                  'epoch', 'brazilian_loss', 'european_loss', 'accuracy', 'f1', 'precision', 'recall', 'domain', 'lr', 'validation_loss'])
        best_f1 = -math.inf

        epochs_without_improvement = 3

        best_val_loss = math.inf

        with torch.enable_grad():
            self.brazilian_model.train()
            self.european_model.train()

            for epoch in range(self.n_epochs):
                if epochs_without_improvement == 0:
                    break
            
                # Create threads for training both models
                brazilian_thread = threading.Thread(target=self.train_model, args=(self.brazilian_model, self.optimizer_brazilian, self.brazilian_train))
                european_thread = threading.Thread(target=self.train_model, args=(self.european_model, self.optimizer_european, self.european_train))

                # Start both threads
                brazilian_thread.start()
                european_thread.start()

                # Wait for both threads to complete
                brazilian_thread.join()
                european_thread.join()
                
                accuracy, f1, precision, recall, validation_loss = self.tester.test(
                    european_model=self.european_model, brazilian_model=self.brazilian_model)

                
                #print(f"Epoch: {epoch} | Brazilian Loss: {brazilian_loss} | European Loss: {european_loss} | Accuracy: {accuracy} | F1: {f1} | Precision: {precision} | Recall: {recall} | Validation Loss: {validation_loss}")

                if f1 > best_f1 and validation_loss < best_val_loss:
                    best_f1 = f1
                    best_val_loss = validation_loss
                    
                    epochs_without_improvement = 3
                    
                    torch.save(self.brazilian_model.state_dict(), os.path.join(
                        CURRENT_PATH, 'out', f'{self.domain}_brazilian_model.pt'))
                    
                    torch.save(self.european_model.state_dict(), os.path.join(
                        CURRENT_PATH, 'out', f'{self.domain}_european_model.pt'))
                    
                    print(f"Saved model with F1: {f1} and Validation Loss: {validation_loss}")

                elif best_f1 > 0.0:
                    print(f"No Improvement. Attempts left: {epochs_without_improvement}")
                    epochs_without_improvement -= 1

                df_results = pd.concat([df_results, pd.DataFrame({
                    'epoch': [epoch],
                    'brazilian_loss': [0],
                    'european_loss': [0],
                    'accuracy': [accuracy],
                    'f1': [f1],
                    'precision': [precision],
                    'recall': [recall],
                    'domain': [self.domain],
                    'validation_loss': [validation_loss],
                })])

        return df_results
