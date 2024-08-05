import os
from multiprocessing import Process
from time import time

import torch
from tqdm import tqdm

from ptvid.constants import DOMAINS
from ptvid.src.bert.data import Data
from ptvid.src.bert.results import Results
from ptvid.src.bert.tester import Tester
from ptvid.src.bert.trainer import Trainer
from ptvid.src.tunning import Tunning
from ptvid.src.utils import create_output_dir, setup_logger


class Run:
    def __init__(self, dataset_name, model_name, batch_size) -> None:
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.current_time = int(time())
        create_output_dir(self.current_path, self.current_time)
        setup_logger(self.current_path, self.current_time)
        self.data = Data(dataset_name, tokenizer_name=model_name, batch_size=batch_size)
        self.model_name = model_name
        tqdm.pandas()

    def tune_with_gpu(self):
        num_gpus = torch.cuda.device_count()

        share_of_data = 0.9 / num_gpus
        processes = []
        for gpu_id in range(num_gpus):
            device = torch.cuda.get_device_name(gpu_id)

            tuner = Tunning(
                data=self.data,
                domains=DOMAINS,
                Results=Results,
                Trainer=Trainer,
                Tester=Tester,
                sample_size=5_000,
                current_path=self.current_path,
                current_time=self.current_time,
                params={
                    "epochs": 30,
                    "early_stoping": 5,
                    "model_name": self.model_name,
                    "device": device,
                },
            )

            process = Process(target=tuner.run, args=(gpu_id * share_of_data, (gpu_id + 1) * share_of_data))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

    def tune_with_cpu(self):
        tuner = Tunning(
            data=self.data,
            domains=DOMAINS,
            Results=Results,
            Trainer=Trainer,
            Tester=Tester,
            sample_size=5_000,
            current_path=self.current_path,
            current_time=self.current_time,
            params={
                "epochs": 30,
                "early_stoping": 5,
                "model_name": self.model_name,
                "device": "cpu",
            },
        )

        tuner.run()

    def tune(self):
        if torch.cuda.is_available():
            return self.tune_with_gpu()
        return self.tune_with_cpu()


if __name__ == "__main__":
    run = Run(
        dataset_name="liaad/PtBrVId",
        model_name="neuralmind/bert-base-portuguese-cased",
        batch_size=64,
    )

    run.tune()
