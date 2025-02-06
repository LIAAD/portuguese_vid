from pydantic import Field
from typing import Optional
from pt_vid.entity.Entity import Entity

class DatasetStats(Entity):
    dataset_name: str
    split: Optional[str] = Field(None, description='The split of the dataset')
    num_docs: int
    num_tokens: int
