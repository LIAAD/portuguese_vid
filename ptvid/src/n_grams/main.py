import json
import os
from time import time

from tqdm import tqdm

from ptvid.src.data import Data
from ptvid.src.n_grams.results import Results
from ptvid.src.n_grams.tester import Tester
from ptvid.src.n_grams.trainer import Trainer
from ptvid.src.tunning import Tunning
from ptvid.src.utils import create_output_dir, setup_logger
from ptvid.constants import DOMAINS


class Run:
    def __init__(self, dataset_name: str, sample_size: int):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.current_time = int(time())
        self.params = self._load_params()

        create_output_dir(self.current_path, self.current_time)
        setup_logger(self.current_path, self.current_time)

        self.data = Data(dataset_name)

        self.tuner = Tunning(
            self.data,
            DOMAINS,
            Results,
            Trainer,
            Tester,
            CURRENT_PATH=self.current_path,
            CURRENT_TIME=self.current_time,
            params=self.params,
            sample_size=sample_size
        )

    def _load_params(self):
        f = open(os.path.join(self.current_path, "in", "params.json"), "r", encoding="utf-8")

        # Fail if params.json does not exist
        if f is None:
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
    runner = Run(dataset_name="liaad/PtBrVId", sample_size=3_000)
    runner.tune()
