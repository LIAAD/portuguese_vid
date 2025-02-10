from pydantic import BaseModel
from pt_vid.entity.TrainingScenario import TrainingScenario

class TrainingResult(BaseModel):
    #f1_macro: float
    #accuracy: float
    p_pos_delexicalization: float
    p_neg_delexicalization: float