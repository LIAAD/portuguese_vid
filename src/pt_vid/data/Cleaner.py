from tqdm import tqdm
from datasets import Dataset
from pt_vid.data.cleaning.CleanupStrategy import CleanupStrategy
from pt_vid.data.cleaning.FastTextLangDetect import FastTextLangDetect
from pt_vid.data.cleaning.DetokenizerStrategy import DetokenizerStrategy
class Cleaner:
    def run(dataset:Dataset):
        for strategy in tqdm([DetokenizerStrategy, CleanupStrategy, FastTextLangDetect]):
            print(f'Running {strategy.__name__}')
            # TODO: Print the number of rows removed
            dataset = strategy.run(dataset)
        
        return dataset