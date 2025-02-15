from pt_vid.entity.Entity import Entity
from pt_vid.Constants import PT_BR_LABEL, PT_PT_LABEL

class InferenceResult(Entity):
    pt_prob: float
    br_prob: float

    def get_prediction(self):
        return PT_BR_LABEL if self.br_prob > self.pt_prob else PT_PT_LABEL