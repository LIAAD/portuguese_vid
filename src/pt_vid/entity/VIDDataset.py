from typing import Optional
from pt_vid.entity.Entity import Entity
from pt_vid.entity.DatasetStats import DatasetStats
from pydantic import model_validator, Field, ConfigDict

class VIDDataset(Entity):    
    dataset: object
    config_name: str = Field(description='The name of the dataset configuration')
    dataset_stats: Optional[DatasetStats] = Field(
        None, 
        description='The statistics of the dataset'
    )
    @model_validator(mode='after')
    def debug_sample_dataset(self):
        self.dataset = self.dataset.shuffle(seed=42).select(range(300))
        return self
    
    @model_validator(mode='after')
    def force_column_names(self):
        # Ensure column text is present
        if 'text' not in self.dataset.column_names:
            raise ValueError('Column text is not present in the dataset')
        
        # Ensure column label is present
        if 'label' not in self.dataset.column_names:
            raise ValueError('Column label is not present in the dataset')

        return self

    @model_validator(mode='after')
    def force_label_type(self):
        unique_labels = set(self.dataset['label'])

        if len(unique_labels) != 2:
            raise ValueError('There should be exactly two unique labels in the dataset')
        
        # Ensure label is either PT-PT or PT-BR
        if 0 not in unique_labels or 1 not in unique_labels:
            raise ValueError('Labels should be PT-PT and PT-BR')

        return self
        
    @model_validator(mode='after')
    def set_dataset_stats(self):
        self.dataset_stats = DatasetStats(
            dataset=self.dataset,
            config_name=self.config_name,
            split=str(self.dataset.split)
        )

        return self