from typing import List
from datasets import Dataset
from nltk.tokenize import word_tokenize
from pt_vid.entity.DatasetStats import DatasetStats
from pt_vid.entity.CorporaStats import CorporaStats as CorporaStatsEntity

class CorporaStats:
    def __init__(self, dataset_stats: List[DatasetStats]):
        self.dataset_stats = dataset_stats

    def _num_docs(self)->int:
        return sum(ds.num_docs for ds in self.dataset_stats)
    
    def _num_tokens(self)->int:
        return sum(ds.num_tokens for ds in self.dataset_stats)
    
    def _min_tokens(self)->int:
        return min(ds.num_tokens for ds in self.dataset_stats)
    
    def _max_tokens(self)->int:
        return max(ds.num_tokens for ds in self.dataset_stats)
    
    def _avg_tokens(self)->float:
        return self._num_tokens() / self._num_docs()
    
    def _std_tokens(self)->float:
        sum_squares = sum((ds.num_tokens - self._avg_tokens()) ** 2 
                        for ds in self.dataset_stats)
        
        return (sum_squares / self._num_docs()) ** 0.5
    

    def run(self)->CorporaStatsEntity:
        return CorporaStatsEntity(
            num_docs=self._num_docs(),
            num_tokens=self._num_tokens(),
            min_tokens=self._min_tokens(),
            max_tokens=self._max_tokens(),
            avg_tokens=self._avg_tokens(),
            std_tokens=self._std_tokens(),
            dataset_stats=self.dataset_stats
        )