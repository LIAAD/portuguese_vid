import os
from multiprocessing import Process
from time import time

import torch
from ptvid.src.bert.data import Data
from ptvid.src.bert.results import Results
from ptvid.src.bert.tester import Tester
from ptvid.src.bert.trainer import Trainer
from ptvid.src.tunning import Tunning
from ptvid.src.utils import create_output_dir, setup_logger
from tqdm import tqdm


class Run:
    def __init__(self, dataset_name, tokenizer_name, model_name, batch_size) -> None:
        self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        self.CURRENT_TIME = int(time())

        create_output_dir(self.CURRENT_PATH, self.CURRENT_TIME)

        setup_logger(self.CURRENT_PATH, self.CURRENT_TIME)

        self.data = Data(dataset_name, tokenizer_name=tokenizer_name, batch_size=batch_size)

        self._DOMAINS = ["literature", "journalistic", "legal", "politics", "web", "social_media"]

        self.model_name = model_name

        tqdm.pandas()

    def tune_with_gpu(self):
        num_gpus = torch.cuda.device_count()

        share_of_data = 0.9 / num_gpus
        processes = []

        for i in range(num_gpus):
            device = torch.cuda.get_device_name(i)

            tuner = Tunning(
                self.data,
                self._DOMAINS,
                Results,
                Trainer,
                Tester,
                5_000,
                self.CURRENT_PATH,
                self.CURRENT_TIME,
                params={
                    "epochs": 30,
                    "early_stoping": 5,
                    "model_name": self.model_name,
                    "device": device,
                },
            )

            process = Process(target=tuner.run, args=(i * share_of_data, (i + 1) * share_of_data))

            processes.append(process)

            process.start()

        for p in processes:
            p.join()

    def tune_with_cpu(self):
        tuner = Tunning(
            self.data,
            self._DOMAINS,
            Results,
            Trainer,
            Tester,
            5_000,
            self.CURRENT_PATH,
            self.CURRENT_TIME,
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
        "arubenruben/portuguese-language-identification-splitted",
        "neuralmind/bert-base-portuguese-cased",
        "neuralmind/bert-base-portuguese-cased",
        20,
    )

    run.tune()
