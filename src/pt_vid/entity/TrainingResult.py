from typing import Optional
from pt_vid.models.Model import Model
from pydantic import BaseModel, Field

class TrainingResult(BaseModel):
    model: Optional[object] = Field(None, description="Object of type Model")
    p_pos: float
    p_ner: float