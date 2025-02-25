from abc import ABC, abstractmethod
from pt_vid.entity.InferenceResult import InferenceResult
class Model(ABC):
    def __init__(self, p_pos: float, p_ner: float):
        super().__init__()
        self.p_pos = p_pos
        self.p_ner = p_ner

    @abstractmethod
    def inference(self, x)->InferenceResult:
        raise NotImplementedError