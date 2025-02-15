from joblib import load
from datasets import load_dataset
from pt_vid.Evaluate import Evaluate

training_results = load('/teamspace/studios/this_studio/portuguese_vid/training_results.joblib')

test_dataset = load_dataset("LCA-PORVID/frmt", split="test")

test_results = []

for training_result in training_results:
    test_result = Evaluate.test(training_result.model, test_dataset)
    test_results.append(test_result)
    