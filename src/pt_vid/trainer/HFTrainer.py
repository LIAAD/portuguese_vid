import torch
from datasets import Dataset
from pt_vid.trainer.Strategy import Strategy
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

print(f"Torch cuda is available: {torch.cuda.is_available()}")

class HFTrainer(Strategy):
    def __init__(self, model_name:str, training_dataset:Dataset, validation_dataset:Dataset=None, eval_dataset:Dataset=None):
        super().__init__(training_dataset, validation_dataset, eval_dataset)
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    
    def _default_training_args(self):
        return TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

    def train(self, training_args:TrainingArguments=None):
        training_args = training_args or self._default_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_dataset,
            eval_dataset=self.validation_dataset or self.eval_dataset
        )

        trainer.train()

        return trainer