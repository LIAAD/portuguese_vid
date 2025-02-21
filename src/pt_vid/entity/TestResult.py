from typing import List
from pt_vid.entity.Entity import Entity

class TestResult(Entity):
    training_result: object
    train_datasets_names: List[str]
    test_dataset_name: str
    p_pos: float
    p_ner: float
    f1_score: float
    accuracy: float