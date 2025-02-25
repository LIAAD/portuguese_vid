import os
from datasets import Dataset
from pt_vid.data.splitters.Strategy import Strategy
from pt_vid.data.splitters.DefaultSplitterStrategy import DefaultSplitterStrategy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
class Splitter:
    def __init__(self, strategy:Strategy=DefaultSplitterStrategy):
        self.strategy = strategy
    
    def run(self, dataset:Dataset, domain:str):
        return self.strategy.run(dataset, domain)