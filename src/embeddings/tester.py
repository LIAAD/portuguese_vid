import evaluate
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class Tester():
    def __init__(self, model, test_dataloader, loss_fn=None):
        self.model = model
        self.test_dataloader = test_dataloader
        self.evaluator_accuracy = evaluate.load("accuracy")
        self.evaluator_f1 = evaluate.load("f1")
        self.evaluator_precision = evaluate.load("precision")
        self.evaluator_recall = evaluate.load("recall")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn

    def test(self):
        total_loss = 0
        all_preds = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Convert Labels from 1D to 2D. Example [4] -> [4x1]
                labels = batch['label'].unsqueeze(1).float().to(self.device)

                pred = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask)

                pred = pred.to(self.device)

                if self.loss_fn:
                    loss = self.loss_fn(pred, labels)
                    total_loss += loss.item()                
                
                pred_probs = torch.sigmoid(pred)

                # Threshold at 0.5 for binary classification
                predicted_labels = (pred_probs > 0.5).flatten().int().cpu().tolist()

                labels = labels.flatten().int().cpu().tolist()

                all_preds.extend(predicted_labels)
                all_labels.extend(labels)

            accuracy = self.evaluator_accuracy.compute(predictions=all_preds, references=all_labels)['accuracy']
            f1 = self.evaluator_f1.compute(predictions=all_preds, references=all_labels)['f1']
            precision = self.evaluator_precision.compute(predictions=all_preds, references=all_labels)['precision']
            recall = self.evaluator_recall.compute(predictions=all_preds, references=all_labels)['recall']
                
            return accuracy, total_loss / len(self.test_dataloader), f1, precision, recall
