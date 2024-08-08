import logging
import os

import numpy as np
from ptvid.constants import DOMAINS


class Tunning:
    def __init__(
        self,
        data,
        domains,
        Results,
        Trainer,
        Tester,
        current_path,
        current_time,
        params=None,
        sample_size: int = None,
    ) -> None:
        self.data = data
        self.Trainer = Trainer
        self.Tester = Tester
        self.sample_size = sample_size
        self.current_path = current_path
        self.current_time = current_time
        self.results = Results(os.path.join(self.current_path, "out", str(current_time)), DOMAINS)
        self.params = params


