from datasets import load_dataset
from pt_vid.Trainer import Trainer
from pt_vid.trainer.HFTrainer import HFTrainer
from pt_vid.trainer.NgramsTrainer import NgramsTrainer

dataset = load_dataset("liaad/PtBrVId", "web")

# Train BERT model

# Train Albertina model
#Trainer(training_strategy=HFTrainer()).train()

# Train N-Grams model


Trainer(training_strategy=NgramsTrainer(training_dataset=dataset['train'].select(range(100)))).train()