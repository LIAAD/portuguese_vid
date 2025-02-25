import os
import json
from pt_vid.data.cleaning.Strategy import Strategy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class SizeBasedFiltering(Strategy):
    THRESHOLDS: dict = json.load(open(os.path.join(CURRENT_DIR, 'thresholds.json'), 'r'))
    
    @staticmethod
    def _run(row, domain):
        if len(row['text']) < SizeBasedFiltering.THRESHOLDS[domain]['min_chars']:
            row['valid'] = False

        elif len(row['text']) > SizeBasedFiltering.THRESHOLDS[domain]['max_chars']:
            row['valid'] = False
        
        else:
            row['valid'] = True

        return row
        

    @staticmethod
    def run(dataset, domain):
        dataset = dataset.map(lambda example: SizeBasedFiltering._run(example, domain))
        
        # Filter out rows that are not valid
        dataset = dataset.filter(lambda example: example['valid'] == True)

        # Remove the valid column
        dataset = dataset.remove_columns(['valid'])

        return dataset