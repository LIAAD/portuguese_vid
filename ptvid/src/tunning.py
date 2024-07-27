import numpy as np
import logging
import os


class Tunning:
    def __init__(
        self, data, domains, Results, Trainer, Tester, sample_size, CURRENT_PATH, CURRENT_TIME, params=None
    ) -> None:
        self.data = data
        self.Trainer = Trainer
        self.Tester = Tester
        self._DOMAINS = domains
        self.sample_size = sample_size
        self.CURRENT_PATH = CURRENT_PATH
        self.CURRENT_TIME = CURRENT_TIME
        self.results = Results(os.path.join(self.CURRENT_PATH, "out", str(CURRENT_TIME)), self._DOMAINS)
        self.params = params

    def run(self, start_pos_prob=0.0, stop_pos_prob=1.0):
        test_dataset = self.data.load_test_set()

        for pos_prob in np.arange(start_pos_prob, stop_pos_prob + 0.1, 0.1):
            for ner_prob in np.arange(0.0, 1.0 + 0.1, 0.1):
                for domain in self._DOMAINS:
                    logging.info(f"Running {domain} pos_prob={pos_prob}, ner_prob={ner_prob}")

                    dataset = self.data.load_domain(
                        domain, balance=True, pos_prob=pos_prob, ner_prob=ner_prob, sample_size=self.sample_size
                    )
                    trainer = self.Trainer(dataset, self.params)
                    results, best_model = trainer.train()
                    test_results, cross_domain_f1 = self.Tester(test_dataset, best_model, train_domain=domain).test()

                    logging.info(f"Cross domain f1 score: {cross_domain_f1} | test_results: {test_results}")
                    self.results.process(
                        cross_domain_f1,
                        domain,
                        test_results,
                        results,
                        balance=True,
                        pos_prob=pos_prob,
                        ner_prob=ner_prob,
                    )
