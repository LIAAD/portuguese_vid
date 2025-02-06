from typing import Optional, List
from pt_vid.entity.Entity import Entity
from pydantic import Field, model_validator
from pt_vid.entity.DatasetStats import DatasetStats

class CorporaStats(Entity):
    num_docs: Optional[int] = Field(
        None, description='The total number of documents in all datasets'
    )
    
    num_tokens: Optional[int] = Field(
        None, description='The total number of tokens in all datasets'
    )

    min_tokens: Optional[int] = Field(
        None, description='The minimum number of tokens in a document'
    )
    
    max_tokens: Optional[int] = Field(
        None, description='The maximum number of tokens in a document'
    )

    avg_tokens: Optional[float] = Field(
        None, description='The average number of tokens in a document'
    )

    std_tokens: Optional[float] = Field(
        None, description='The standard deviation of the number of tokens in a document'
    )
    
    dataset_stats: List[DatasetStats] = Field(
        description='The statistics of each dataset'
    )