from tqdm import tqdm
from pt_vid.entity.TestResult import TestResult
from sklearn.metrics import accuracy_score, f1_score
from pt_vid.entity.TrainingResult import TrainingResult

class Evaluate:
    @staticmethod    
    def test(training_result:TrainingResult, test_dataset, test_dataset_name: str) -> TestResult:
        y_true = []
        y_pred = []

        #TODO: Optimize using batch inference
        #TODO: Optimize using .map()
        for row in tqdm(test_dataset, desc="Testing"):
            y_true.append(row['label'])
            y_pred.append(training_result.model.inference(row['text']).get_prediction())

        return TestResult(
            training_result=training_result,
            train_datasets_names=training_result.training_datasets_names,
            test_dataset_name=test_dataset_name,
            p_pos=training_result.model.p_pos,
            p_ner=training_result.model.p_ner,
            f1_score=f1_score(y_true, y_pred, average='micro'),
            accuracy=accuracy_score(y_true, y_pred)
        )