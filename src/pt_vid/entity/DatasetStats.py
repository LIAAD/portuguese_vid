from typing import Optional
from datasets import Dataset
from pt_vid.entity.Entity import Entity
from nltk.tokenize import word_tokenize
from pydantic import Field, model_validator

class DatasetStats(Entity):
    dataset: Dataset
    config_name: str = Field(description='The name of the dataset configuration')
    split: str = Field(description='The split of the dataset')
    num_docs: Optional[int] = Field(None, description='The number of documents in the dataset')
    num_tokens: Optional[int] = Field(None, description='The number of tokens in the dataset')

    # TODO: Logic to obtain these stats should be implemented here
    @model_validator(mode='after')
    def set_num_docs(self):
        self.num_docs = len(self.dataset)
    
    @model_validator(mode='after')
    def set_num_tokens(self):
        self.num_tokens = sum([len(word_tokenize(example["text"], language="portuguese")) for example in self.dataset]) 
       