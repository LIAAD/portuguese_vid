import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from pt_vid.data.Tokenizer import Tokenizer
from pt_vid.trainer.Strategy import Strategy
from pt_vid.data.Delexicalizer import Delexicalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pt_vid.entity.NgramsTrainingResult import NgramsTrainingResult
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from pt_vid.entity.NgramsTrainingScenario import NgramsTrainingScenario

if not nltk.download('stopwords'):
    nltk.download('stopwords')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class NgramsTrainer(Strategy):
    def __init__(self, training_dataset, validation_dataset = None, eval_dataset = None, parameters_filepath=None, *args, **kwargs):
        super().__init__(training_dataset, validation_dataset, eval_dataset, *args, **kwargs)        
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
                tokenizer=lambda x: Tokenizer().tokenize(x),
                stop_words=nltk.corpus.stopwords.words("portuguese"),
                token_pattern=None
            )),
            ("classifier", BernoulliNB())
        ])
        
        self.cv = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)
        
        self.search = RandomizedSearchCV(
            self.pipeline,
            self.sklearn_parameters,
            scoring="f1_macro",
            n_jobs=2,
            #n_iter=500,
            cv=self.cv,
            error_score="raise",
            verbose=10
        )
        
    def train(self):
        results = []

        text = self.training_dataset["text"]
        labels = self.training_dataset["label"]
        
        for p_pos_delexicalization in tqdm([0, 0.25, 0.5, 0.75, 1]):
            for p_neg_delexicalization in tqdm([0, 0.25, 0.5, 0.75, 1]):
                delexicalizer = Delexicalizer(
                    prob_ner_tag=p_neg_delexicalization, 
                    prob_pos_tag=p_pos_delexicalization,
                )
                
                new_text = [delexicalizer.delexicalize(t) for t in text]

                result = self.search.fit(np.array(new_text), np.array(labels))

                results.append(NgramsTrainingResult(
                    best_pipeline=result.best_estimator_,
                    best_tf_idf_max_features=result.best_params_["tf_idf__max_features"],
                    best_tf_idf_ngram_range=result.best_params_["tf_idf__ngram_range"],
                    best_tf_idf_lower_case=result.best_params_["tf_idf__lowercase"],
                    best_tf_idf_analyzer=result.best_params_["tf_idf__analyzer"],
                    p_pos_delexicalization=p_pos_delexicalization,
                    p_neg_delexicalization=p_neg_delexicalization,
                ))
                
        return results