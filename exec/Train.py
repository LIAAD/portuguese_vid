from environs import Env
from joblib import dump, load
from pt_vid.Trainer import Trainer
from pt_vid.trainer.HFTrainer import HFTrainer
from pt_vid.trainer.NgramsTrainer import NgramsTrainer
from datasets import load_dataset, concatenate_datasets

env = Env()
env.read_env(override=True)

dataset = load_dataset("liaad/PtBrVId", "web")

# Train BERT model

# Train Albertina model
#Trainer(training_strategy=HFTrainer()).train()

if env.bool('DEBUG', False):
    # Extract 50 examples from the training dataset with label 0 and 50 examples with label 1
    training_dataset = dataset['train'].shuffle(seed=42)
    training_dataset_with_label_0 = training_dataset.filter(lambda example: example['label'] == 0).select(range(50))
    training_dataset_with_label_1 = training_dataset.filter(lambda example: example['label'] == 1).select(range(50))
    #training_dataset = training_dataset_with_label_0.concatenate(training_dataset_with_label_1)
    training_dataset = concatenate_datasets([training_dataset_with_label_0, training_dataset_with_label_1])
else:
    training_dataset = dataset['train']

# Train N-Grams model
training_results = Trainer(
    training_strategy=NgramsTrainer(
        training_dataset=training_dataset
)).train()

pass