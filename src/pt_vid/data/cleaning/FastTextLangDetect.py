from ftlangdetect import detect
from pt_vid.data.cleaning.Strategy import Strategy
class FastTextLangDetect(Strategy):

    FAST_TEXT_THRESHOLD = 0.9

    def _run(row):
        result = detect(row['text'])

        if result['lang'] == 'pt' and result['score'] >= FastTextLangDetect.FAST_TEXT_THRESHOLD:
            row['lang'] = 'pt'
        else:
            row['lang'] = "other"

        return row
        
    def run(dataset, domain):
        dataset = dataset.map(lambda example: FastTextLangDetect._run(example))
        # Filter out rows that are not in Portuguese
        dataset = dataset.filter(lambda example: example['lang'] == 'pt')

        # Remove the lang column
        dataset = dataset.remove_columns(['lang'])
        
        return dataset