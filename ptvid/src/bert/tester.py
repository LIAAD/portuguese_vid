import torch
import evaluate
from tqdm import tqdm
import logging


class Tester:
    def __init__(self, test_dataset_dict, model, train_domain) -> None:
        self.test_dataset_dict = test_dataset_dict
        self.model = model
        self.train_domain = train_domain

        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.loss_fn = torch.nn.BCELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _test(self, test_dataset):
        with torch.no_grad():
            total_loss = 0

            for batch in tqdm(test_dataset):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask).squeeze(dim=1)

                loss = self.loss_fn(logits, labels.float())

                # If logits is bigger than 0.5, it's 1, otherwise it's 0
                predictions = (logits > 0.5).long()

                # Detach from GPU
                predictions = predictions.cpu()
                labels = labels.cpu()

                accuracy = self.accuracy.add_batch(predictions=predictions, references=labels)

                f1 = self.f1.add_batch(predictions=predictions, references=labels)

                precision = self.precision.add_batch(predictions=predictions, references=labels)

                recall = self.recall.add_batch(predictions=predictions, references=labels)

                total_loss += loss.item()

            accuracy = self.accuracy.compute()["accuracy"]
            f1 = self.f1.compute()["f1"]
            precision = self.precision.compute()["precision"]
            recall = self.recall.compute()["recall"]
            total_loss = total_loss / len(test_dataset)

            return accuracy, f1, precision, recall, total_loss

    def test(self):
        self.model.eval()
        self.model.to(self.device)

        results = {}

        for domain in self.test_dataset_dict.keys():
            logging.info(f"Testing {domain} domain...")
            accuracy, f1, precision, recall, total_loss = self._test(self.test_dataset_dict[domain])

            results[domain] = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "loss": total_loss,
            }

        # Calculate the average of all domains except the train domain
        average_f1 = sum([results[domain]["f1"] for domain in results.keys() if domain != self.train_domain]) / (
            len(results.keys()) - 1
        )

        return results, average_f1
