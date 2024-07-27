import json
import os
from time import time

from pt_variety_identifier.src.data import Data
from pt_variety_identifier.src.n_grams.results import Results
from pt_variety_identifier.src.n_grams.tester import Tester
from pt_variety_identifier.src.n_grams.trainer import Trainer
from pt_variety_identifier.src.tunning import Tunning
from pt_variety_identifier.src.utils import create_output_dir, setup_logger
from tqdm import tqdm


class Run:
    def __init__(self, dataset_name) -> None:
        self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        self.CURRENT_TIME = int(time())
        self.params = self.load_params()

        create_output_dir(self.CURRENT_PATH, self.CURRENT_TIME)
        setup_logger(self.CURRENT_PATH, self.CURRENT_TIME)

        self.data = Data(dataset_name)
        self._DOMAINS = ["literature", "journalistic", "legal", "politics", "web", "social_media"]

        # Enable progress bar for pandas
        tqdm.pandas()

        self.tuner = Tunning(
            self.data,
            self._DOMAINS,
            Results,
            Trainer,
            Tester,
            sample_size=5_000,
            CURRENT_PATH=self.CURRENT_PATH,
            CURRENT_TIME=self.CURRENT_TIME,
            params=self.params,
        )

    def load_params(self):
        f = open(os.path.join(self.CURRENT_PATH, "in", "params.json"), "r", encoding="utf-8")

        # Fail if params.json does not exist
        if f == None:
            raise FileNotFoundError("params.json not found")

        dict_obj = json.load(f)

        if "tfidf__ngram_range" in dict_obj:
            # Cast tfidf__ngram_range to tuple
            for idx, elem in enumerate(dict_obj["tfidf__ngram_range"]):
                dict_obj["tfidf__ngram_range"][idx] = tuple(elem)

        return dict_obj

    def tune(self):
        return self.tuner.run()


if __name__ == "__main__":
    runner = Run(dataset_name="arubenruben/portuguese-language-identification-splitted")

    runner.tune()
