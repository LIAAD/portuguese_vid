from typing import Literal
from pydantic import model_validator
from pt_vid.models.Ngrams import Ngrams
from pt_vid.entity.TrainingResult import TrainingResult
class NgramsTrainingResult(TrainingResult):
    best_tf_idf_max_features: int
    best_tf_idf_ngram_range: tuple[int, int]
    best_tf_idf_lower_case: bool
    best_tf_idf_analyzer: Literal["word", "char"]
    mean_f1_train: float
    mean_accuracy_train: float
    best_f1_train: float
    best_pipeline: object

    @model_validator(mode="after")
    def set_model(self):
        self.model = Ngrams(
            p_pos=self.p_pos,
            p_ner=self.p_ner,
            pipeline=self.best_pipeline,
        )

        return self
