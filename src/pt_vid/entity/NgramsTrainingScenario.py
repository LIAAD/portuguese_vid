from typing import Literal
from pydantic import BaseModel
from pt_vid.entity.TrainingScenario import TrainingScenario

class NgramsTrainingScenario(TrainingScenario):
    tf_idf_max_features: int
    tf_idf_ngram_range: tuple[int, int]
    tf_idf_lower_case: bool
    tf_idf_analyzer: Literal["word", "char"]

    
    @staticmethod
    def concatenate_dumps(dumps):
        return {
            'tf_idf__max_features': [dump.tf_idf_max_features for dump in dumps],
            'tf_idf__ngram_range': [dump.tf_idf_ngram_range for dump in dumps],
            'tf_idf__lowercase': [dump.tf_idf_lower_case for dump in dumps],
            'tf_idf__analyzer': [dump.tf_idf_analyzer for dump in dumps]
        }