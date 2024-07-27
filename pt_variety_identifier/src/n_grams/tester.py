import evaluate


class Tester:
    def __init__(self, test_dataset_dict, pipeline, train_domain) -> None:
        self.test_dataset_dict = test_dataset_dict
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.pipeline = pipeline
        self.train_domain = train_domain

    def _test(self):
        predictions = self.pipeline.predict(self.test_dataset['text'])

        accuracy = self.accuracy.compute(
            references=self.test_dataset['label'], predictions=predictions)['accuracy']

        f1 = self.f1.compute(
            references=self.test_dataset['label'], predictions=predictions)['f1']

        precision = self.precision.compute(
            references=self.test_dataset['label'], predictions=predictions)['precision']

        recall = self.recall.compute(
            references=self.test_dataset['label'], predictions=predictions)['recall']

        return accuracy, f1, precision, recall

    def test(self):
        results = {}

        for domain in self.test_dataset_dict.keys():
            self.test_dataset = self.test_dataset_dict[domain]

            accuracy, f1, precision, recall = self._test()

            results[domain] = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
        # Calculate the average of all domains except the train domain
        average_f1 = sum([results[domain]['f1'] for domain in results.keys() if domain != self.train_domain]) / (len(results.keys()) - 1)

        return results, average_f1