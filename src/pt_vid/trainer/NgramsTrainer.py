import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from pt_vid.data.Tokenizer import Tokenizer
from sklearn.naive_bayes import BernoulliNB
from pt_vid.trainer.Strategy import Strategy
from pt_vid.data.Delexicalizer import Delexicalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from pt_vid.entity.NgramsTrainingResult import NgramsTrainingResult
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from pt_vid.entity.NgramsTrainingScenario import NgramsTrainingScenario

if not nltk.download('stopwords'):
    nltk.download('stopwords')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def tokenizer(text):
    return Tokenizer.tokenize(text)
class NgramsTrainer(Strategy):
    def __init__(self, 
                 training_dataset,
                 training_datasets_names,
                 parameters_filepath=None, 
                 scoring=None,
                 *args, **kwargs):
        super().__init__(training_dataset, training_datasets_names, *args, **kwargs)        
        self.parameters_filepath = parameters_filepath or os.path.join(CURRENT_DIR, 'ngrams_scenarios.json')
        
        self.parameters = []
        
        for key in json.load(open(self.parameters_filepath)):
            self.parameters.append(NgramsTrainingScenario(**{
                'name': key,
                **json.load(open(self.parameters_filepath))[key]
            }))

        self.sklearn_parameters = NgramsTrainingScenario.concatenate_dumps(self.parameters)
                
        self.pipeline = Pipeline([
            ("tf_idf", TfidfVectorizer(
                tokenizer=tokenizer,
                stop_words=nltk.corpus.stopwords.words("portuguese"),
                token_pattern=None
            )),
            ("classifier", BernoulliNB())
        ])
        
        self.cv = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)
        
        scoring = {
            'f1': make_scorer(f1_score),
            'accuracy': make_scorer(accuracy_score)
        }
        
        self.search = RandomizedSearchCV(
            self.pipeline,
            self.sklearn_parameters,
            scoring=scoring,
            refit='f1', # Use F1 score to select the best model
            n_jobs=2,            
            cv=self.cv,
            error_score="raise",
            verbose=10,
            return_train_score=True,
        )
        
    def train(self):
        results = []

        text = self.training_dataset["text"]
        labels = self.training_dataset["label"]
        
        for p_pos in tqdm([0, 0.25, 0.5, 0.75, 1]):
            for p_ner in tqdm([0, 0.25, 0.5, 0.75, 1]):
                delexicalizer = Delexicalizer(
                    prob_ner_tag=p_ner, 
                    prob_pos_tag=p_pos,
                )
                
                new_text = [delexicalizer.delexicalize(t) for t in text]

                result = self.search.fit(np.array(new_text), np.array(labels))

                results.append(NgramsTrainingResult(
                    best_pipeline=result.best_estimator_,
                    best_tf_idf_max_features=result.best_params_["tf_idf__max_features"],
                    best_tf_idf_ngram_range=result.best_params_["tf_idf__ngram_range"],
                    best_tf_idf_lower_case=result.best_params_["tf_idf__lowercase"],
                    best_tf_idf_analyzer=result.best_params_["tf_idf__analyzer"],
                    p_pos=p_pos,
                    p_ner=p_ner,
                    mean_f1_train=result.cv_results_["mean_test_f1"].mean(),
                    mean_accuracy_train=self.search.cv_results_["mean_test_accuracy"].mean(),
                    best_f1_train=result.best_score_,
                    training_datasets_names=self.training_datasets_names
                ))
                
        return results