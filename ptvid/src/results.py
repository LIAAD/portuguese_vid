import json
import math
import os


class Results:
    def __init__(self, filepath, DOMAINS) -> None:
        self.filepath = filepath
        self.best_intermediate_results_list = []
        self.other_results_list = []
        self.DOMAINS = DOMAINS

        self.best_f1_scores = {
            domain: {"cross_domain_f1": -math.inf, "params": {}, "balance": None, "pos_prob": None, "ner_prob": None}
            for domain in self.DOMAINS
        }

    def best_intermediate_results(self, result):
        self.best_intermediate_results_list.append(result)

        with open(os.path.join(self.filepath, "best_intermediate_self.json"), "w", encoding="utf-8") as f:
            json.dump(self.best_intermediate_results_list, f, ensure_ascii=False, indent=4)

    def best_final_results(self):
        with open(os.path.join(self.filepath, "best_final_self.json"), "w", encoding="utf-8") as f:
            json.dump(self.best_f1_scores, f, ensure_ascii=False, indent=4)

    def other_results(self, result):
        self.other_results_list.append(result)

        with open(os.path.join(self.filepath, "other_self.json"), "w", encoding="utf-8") as f:
            json.dump(self.other_results_list, f, ensure_ascii=False, indent=4)
