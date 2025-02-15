from tqdm import tqdm
from pt_vid.models.Model import Model
from pt_vid.entity.TestResult import TestResult
from sklearn.metrics import accuracy_score, f1_score

class Evaluate:
    @staticmethod    
    def test(model: Model, test_dataset):
        y_true = []
        y_pred = []

        #TODO: Optimize using batch inference
        #TODO: Optimize using .map()
        for row in tqdm(test_dataset, desc="Testing"):
            y_true.append(row['label'])
            y_pred.append(model.inference(row['text']).get_prediction())

        return TestResult(
            model=model,
            p_pos=model.p_pos,
            p_ner=model.p_ner,
            f1_score=f1_score(y_true, y_pred, average='micro'),
            accuracy=accuracy_score(y_true, y_pred)
        )