from typing import Optional
from datasets import Dataset
from pt_vid.entity.Entity import Entity
from pydantic import model_validator, Field
from pt_vid.entity.DatasetStats import DatasetStats

class VIDDataset(Entity):
    raw_dataset: Dataset
    dataset_stats: Optional[DatasetStats] = Field(
        None, 
        description='The statistics of the dataset'
    )

    @model_validator(mode='after')
    def force_column_names(self):
        # Ensure column text is present
        if 'text' not in self.raw_dataset.column_names:
            raise ValueError('Column text is not present in the dataset')
        
        # Ensure column label is present
        if 'label' not in self.raw_dataset.column_names:
            raise ValueError('Column label is not present in the dataset')

    @model_validator(mode='after')
    def force_label_type(self):
        unique_labels = self.raw_dataset['label'].unique()

        if len(unique_labels) != 2:
            raise ValueError('There should be exactly two unique labels in the dataset')
        
        # Ensure label is either PT-PT or PT-BR
        if 'PT-PT' not in unique_labels or 'PT-BR' not in unique_labels:
            raise ValueError('Labels should be PT-PT and PT-BR')
        
    @model_validator(mode='after')
    def set_dataset_stats(self):
        self.dataset_stats = DatasetStats(
            dataset=self.raw_dataset,
            config_name=self.raw_dataset.config_name,
            split=self.raw_dataset.split
        )