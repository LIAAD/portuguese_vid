from pt_vid.models.Model import Model
from pt_vid.entity.InferenceResult import InferenceResult
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Transformer(Model):
    def __init__(self, p_pos, p_ner, model_name):
        super().__init__(p_pos, p_ner)
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, x)->InferenceResult:
        raise NotImplementedError