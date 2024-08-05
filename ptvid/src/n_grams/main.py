import json
import os
from time import time

from ptvid.constants import DOMAINS, SAMPLE_SIZE
from ptvid.src.data import Data
from ptvid.src.n_grams.results import Results
from ptvid.src.n_grams.tester import Tester
from ptvid.src.n_grams.trainer import Trainer
from ptvid.src.tunning import Tunning
from ptvid.src.utils import create_output_dir, setup_logger


def run(dataset_name: str, sample_size: int):
    current_path = os.path.dirname(os.path.abspath(__file__))
    current_time = int(time())
    params = _load_params(current_path)

    create_output_dir(current_path, current_time)
    setup_logger(current_path, current_time)

    data = Data(dataset_name)

    tuner = Tunning(
        data,
        DOMAINS,
        Results,
        Trainer,
        Tester,
        current_path=current_path,
        current_time=current_time,
        params=params,
        sample_size=sample_size,
    )

    return tuner.run()


def _load_params(current_path: str):
    params_file_path = os.path.join(current_path, "in", "params.json")

    with open(params_file_path, "r", encoding="utf-8") as f:
        dict_obj = json.load(f)

    if "tfidf__ngram_range" in dict_obj:
        # Cast tfidf__ngram_range to tuple
        for idx, elem in enumerate(dict_obj["tfidf__ngram_range"]):
            dict_obj["tfidf__ngram_range"][idx] = tuple(elem)

    return dict_obj


if __name__ == "__main__":
    run(dataset_name="liaad/PtBrVId", sample_size=SAMPLE_SIZE)
