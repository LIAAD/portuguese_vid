from tqdm import tqdm
from datasets import Dataset
from pt_vid.data.cleaning.CleanupStrategy import CleanupStrategy
from pt_vid.data.cleaning.FastTextLangDetect import FastTextLangDetect
from pt_vid.data.cleaning.SizeBasedFiltering import SizeBasedFiltering
from pt_vid.data.cleaning.DetokenizerStrategy import DetokenizerStrategy
class Cleaner:
    def run(dataset:Dataset, domain:str=None):
        for strategy in tqdm([DetokenizerStrategy, CleanupStrategy, FastTextLangDetect, SizeBasedFiltering]):
            print(f'Running {strategy.__name__}')
            # TODO: Print the number of rows removed
            dataset = strategy.run(dataset, domain)
        
        return dataset