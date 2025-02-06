from typing import Optional, List
from pt_vid.entity.Entity import Entity
from pydantic import Field, model_validator
from pt_vid.entity.DatasetStats import DatasetStats

class Stat(Entity):
    config_name: str = Field(description='The name of the dataset configuration')
    value: int = Field(description='The value of the statistic')

class CorporaStats(Entity):
    num_docs: Optional[int] = Field(
        None, description='The total number of documents in all datasets'
    )
    
    num_tokens: Optional[int] = Field(
        None, description='The total number of tokens in all datasets'
    )

    min_tokens: Optional[Stat] = Field(
        None, description='The minimum number of tokens in a document'
    )
    
    max_tokens: Optional[Stat] = Field(
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
    def set_num_docs(self):
        self.num_docs = sum([ds.num_docs for ds in self.dataset_stats])
        return self
    
    @model_validator(mode='after')
    def set_num_tokens(self):
        self.num_tokens = sum([ds.num_tokens for ds in self.dataset_stats])
        return self
    
    @model_validator(mode='after')
    def set_min_tokens(self):
        dataset_with_min_tokens = min(self.dataset_stats, key=lambda x: x.num_tokens)
        
        self.min_tokens = Stat(
            config_name=dataset_with_min_tokens.config_name,
            value=dataset_with_min_tokens.num_tokens
        )

        return self
    
    @model_validator(mode='after')
    def set_max_tokens(self):
        dataset_with_max_tokens = max(self.dataset_stats, key=lambda x: x.num_tokens)
        
        self.max_tokens = Stat(
            config_name=dataset_with_max_tokens.config_name,
            value=dataset_with_max_tokens.num_tokens
        )

        return self

    @model_validator(mode='after')
    def set_avg_tokens(self):
        self.avg_tokens = sum([ds.avg_tokens for ds in self.dataset_stats]) / len(self.dataset_stats)
        
        return self

    @model_validator(mode='after')
    def set_std_tokens(self):
        self.std_tokens = sum([ds.std_tokens for ds in self.dataset_stats]) / len(self.dataset_stats)
        return self