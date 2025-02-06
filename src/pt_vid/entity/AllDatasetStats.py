from typing import Optional, List
from pt_vid.entity.Entity import Entity
from pydantic import Field, model_validator
from pt_vid.entity.DatasetStats import DatasetStats

class AllDatasetStats(Entity):
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

    @model_validator(mode='after')
    def fill_num_docs(cls, values):
        if values.num_docs is None:
            values.num_docs = sum(ds.num_docs for ds in values.dataset_stats)
        
        return values

    @model_validator(mode='after')
    def fill_num_tokens(cls, values):
        if values.num_tokens is None:
            values.num_tokens = sum(ds.num_tokens for ds in values.dataset_stats)
        
        return values

    @model_validator(mode='after')
    def fill_min_tokens(cls, values):
        if values.min_tokens is None:
            values.min_tokens = min(ds.num_tokens for ds in values.dataset_stats)
        
        return values

    @model_validator(mode='after')
    def fill_max_tokens(cls, values):
        if values.max_tokens is None:
            values.max_tokens = max(ds.num_tokens for ds in values.dataset_stats)
        
        return values

    @model_validator(mode='after')
    def fill_avg_tokens(cls, values):
        if values.avg_tokens is None and values.num_docs:
            values.avg_tokens = values.num_tokens / values.num_docs
        
        return values

    @model_validator(mode='after')
    def fill_std_tokens(cls, values):
        if values.std_tokens is None and values.num_docs and values.avg_tokens:
            sum_squares = sum((ds.num_tokens - values.avg_tokens) ** 2 
                            for ds in values.dataset_stats)
            values.std_tokens = (sum_squares / values.num_docs) ** 0.5
        
        return values