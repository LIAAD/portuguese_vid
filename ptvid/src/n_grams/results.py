import logging
import os

import pandas as pd
from joblib import dump
from pt_variety_identifier.src.results import Results as BaseResults


class Results(BaseResults):
    def __init__(self, filepath, DOMAINS) -> None:
        super().__init__(filepath, DOMAINS)

    def process(self, cross_domain_f1, train_domain, test_results, grid_results, balance, pos_prob, ner_prob):
        if cross_domain_f1 > self.best_f1_scores[train_domain]["cross_domain_f1"]:
            logging.info(f"New best f1 score for {train_domain}")

            self.best_f1_scores[train_domain]["cross_domain_f1"] = cross_domain_f1
            self.best_f1_scores[train_domain]["test_results"] = test_results
            self.best_f1_scores[train_domain]["params"] = grid_results.best_params_
            self.best_f1_scores[train_domain]["balance"] = balance
            self.best_f1_scores[train_domain]["pos_prob"] = pos_prob
            self.best_f1_scores[train_domain]["ner_prob"] = ner_prob

            logging.info(f"Saving best cross_domain_f1 scores to file")

            self.best_final_results()

            with open(os.path.join(self.filepath, "models", f"{train_domain}.joblib"), "wb") as f:
                dump(grid_results.best_estimator_, f)

        self.best_intermediate_results(
            {
                "domain": train_domain,
                "balance": balance,
                "pos_prob": pos_prob,
                "ner_prob": ner_prob,
                "train": {
                    "best_score": grid_results.best_score_,
                },
                "test": {"all": test_results, "cross_domain_f1": cross_domain_f1},
                "best_params": grid_results.best_params_,
            }
        )

        self.other_results(
            {
                "domain": train_domain,
                "balance": balance,
                "pos_prob": pos_prob,
                "ner_prob": ner_prob,
                "train": {"cv_results": pd.DataFrame(grid_results.cv_results_).to_json()},
                "test": test_results,
            }
        )
