import nltk
from typing import Optional
from pt_vid.entity.Entity import Entity
from nltk.tokenize import word_tokenize
from pydantic import Field, model_validator

nltk.download('punkt_tab')

class DatasetStats(Entity):
    dataset: object
    config_name: Optional[str] = Field(None, description='The name of the dataset configuration')
    split: str = Field(description='The split of the dataset')
    num_docs: Optional[int] = Field(None, description='The number of documents in the dataset')
    num_tokens: Optional[int] = Field(None, description='The number of tokens in the dataset')
    avg_tokens: Optional[float] = Field(None, description='The average number of tokens in a document')
    std_tokens: Optional[float] = Field(None, description='The standard deviation of the number of tokens in a document')

    # TODO: Logic to obtain these stats should be implemented here
    @model_validator(mode='after')
    def set_num_docs(self):
        self.num_docs = len(self.dataset)
        
        return self

    def _set_num_tokens(self, document):
        document['num_tokens'] = len(word_tokenize(document['text'], language="portuguese"))
        return document
    
    @model_validator(mode='after')
    def set_num_tokens(self):
        
        self.dataset = self.dataset.map(self._set_num_tokens)

        self.num_tokens = sum(self.dataset['num_tokens'])
        
        return self
    
    @model_validator(mode='after')
    def set_avg_tokens(self):
        import numpy as np
        self.avg_tokens = np.mean(self.dataset['num_tokens'])
        
        return self
    
    @model_validator(mode='after')
    def set_std_tokens(self):
        import numpy as np
        self.std_tokens = np.std(self.dataset['num_tokens'])
        
        return self
    
    # This method should be the last one to be called
    @model_validator(mode='after')
    def remove_num_tokens(self):
        self.dataset.remove_columns('num_tokens')
        return self