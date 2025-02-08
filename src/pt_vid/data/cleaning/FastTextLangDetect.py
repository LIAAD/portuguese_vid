from ftlangdetect import detect
from pt_vid.data.cleaning.Strategy import Strategy

class FastTextLangDetect(Strategy):

    FAST_TEXT_THRESHOLD = 0.9

    def _run(text):
        return detect(text)['lang'] == 'pt' and detect(text)['score'] >= FastTextLangDetect.FAST_TEXT_THRESHOLD
    
    def run(dataset):
        for i in range(len(dataset)):
            if not FastTextLangDetect._run(dataset[i]['text']):
                dataset.remove(i)
        return dataset