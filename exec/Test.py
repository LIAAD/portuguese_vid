from joblib import load
from pt_vid.Plot import Plot
from datasets import load_dataset
from pt_vid.Evaluate import Evaluate

training_results = load('/teamspace/studios/this_studio/training_results.joblib')

test_dataset = load_dataset("LCA-PORVID/frmt", split="test")

test_results = []

for training_result in training_results:
    test_result = Evaluate.test(training_result, test_dataset, "LCA-PORVID/frmt")
    test_results.append(test_result)

Plot.heatmap(test_results)
    