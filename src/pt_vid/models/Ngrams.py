from sklearn.pipeline import Pipeline
from pt_vid.models.Model import Model
from pt_vid.entity.InferenceResult import InferenceResult

class Ngrams(Model):
    def __init__(self, p_pos: float, p_ner: float, pipeline: Pipeline):
        super().__init__(p_pos, p_ner)
        self.pipeline = pipeline
    
    def inference(self, x:str)->InferenceResult:
        results_proba = self.pipeline.predict_proba([x])
        
        return InferenceResult(
            br_prob=results_proba[0][1],
            pt_prob=results_proba[0][0]
        )