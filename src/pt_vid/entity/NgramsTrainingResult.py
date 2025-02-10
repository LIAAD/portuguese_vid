from typing import Literal
from sklearn.pipeline import Pipeline
from pt_vid.entity.TrainingResult import TrainingResult

class NgramsTrainingResult(TrainingResult):
    best_tf_idf_max_features: int
    best_tf_idf_ngram_range: tuple[int, int]
    best_tf_idf_lower_case: bool
    best_tf_idf_analyzer: Literal["word", "char"]
    best_pipeline: object