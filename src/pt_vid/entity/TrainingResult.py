from pydantic import BaseModel

class TrainingResult(BaseModel):
    model: object
    p_pos: float
    p_ner: float