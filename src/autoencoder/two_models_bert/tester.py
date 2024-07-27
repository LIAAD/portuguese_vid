import torch
from tqdm import tqdm
import evaluate
from data import load_test_data
from pathlib import Path

CURRENT_PATH = Path(__file__).parent


class Tester():
    def __init__(self, domain):
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.test_dataloader = load_test_data(domain=domain)
        
        self.reconstruction_loss = torch.nn.MSELoss(reduction='none')

    def test(self, european_model, brazilian_model):
        european_model.eval()
        brazilian_model.eval()
        
        predictions = []
        labels = []

        total_loss = 0
        
        with torch.no_grad():

            for batch in tqdm(self.test_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                
                attention_mask = batch['attention_mask'].to(self.device)
                
                label = batch['label'].to(self.device)

                bert_european, reconstruction_european = european_model(input_ids=input_ids, attention_mask=attention_mask)
                bert_brazilian, reconstruction_brazilian = brazilian_model(input_ids=input_ids, attention_mask=attention_mask)

                test_loss_european = self.reconstruction_loss(
                    reconstruction_european, bert_european)
                
                test_loss_brazilian = self.reconstruction_loss(
                    reconstruction_brazilian, bert_brazilian)
                
                for loss_european, loss_brazilian in zip(test_loss_european, test_loss_brazilian):
                    
                    if loss_european.mean().item() < loss_brazilian.mean().item():
                        predictions.append(0)
                        total_loss += loss_european.mean().item() / len(test_loss_european)
                    
                    else:
                        predictions.append(1)
                        total_loss += loss_brazilian.mean().item() / len(test_loss_brazilian)
                
                labels.extend(label.tolist())
                

        accuracy = self.accuracy.compute(predictions=predictions, references=labels)['accuracy']
        f1 = self.f1.compute(predictions=predictions, references=labels)['f1']
        precision = self.precision.compute(predictions=predictions, references=labels)['precision']
        recall = self.recall.compute(predictions=predictions, references=labels)['recall']

        return accuracy, f1, precision, recall, total_loss / len(self.test_dataloader)

            
