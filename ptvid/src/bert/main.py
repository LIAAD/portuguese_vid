import os
from time import time

import torch
from tqdm import tqdm

from ptvid.constants import DOMAINS, SAMPLE_SIZE
from ptvid.src.bert.data import Data
from ptvid.src.bert.results import Results
from ptvid.src.bert.tester import Tester
from ptvid.src.bert.trainer import Trainer
from ptvid.src.tunning import Tunning
from ptvid.src.utils import create_output_dir, setup_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(dataset_name, model_name, batch_size):
    current_path = os.path.dirname(os.path.abspath(__file__))
    current_time = int(time())
    create_output_dir(current_path, current_time)
    setup_logger(current_path, current_time)
    
    data = Data(dataset_name, tokenizer_name=model_name, batch_size=batch_size)
    tqdm.pandas()
    
    tuner = Tunning(
        data=data,
        domains=DOMAINS,
        Results=Results,
        Trainer=Trainer,
        Tester=Tester,
        sample_size=SAMPLE_SIZE,
        current_path=current_path,
        current_time=current_time,
        params={
            "epochs": 30,
            "early_stoping": 5,
            "model_name": model_name,
            "device": DEVICE,
        },
    )
    tuner.run()

if __name__ == "__main__":
    main(
        dataset_name="liaad/PtBrVId",
        model_name="neuralmind/bert-base-portuguese-cased",
        batch_size=64,
    )
