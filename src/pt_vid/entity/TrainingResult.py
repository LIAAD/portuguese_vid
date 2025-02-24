from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class TrainingResult(BaseModel):
    model: Optional[object] = Field(None, description="Object of type Model")
    training_datasets_names: List[str]
    p_pos: float
    p_ner: float