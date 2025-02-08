from pt_vid.data.cleaning.CleanupStrategy import CleanupStrategy
from pt_vid.data.cleaning.FastTextLangDetect import FastTextLangDetect
from pt_vid.data.cleaning.DetokenizerStrategy import DetokenizerStrategy

class Cleaner:
    def run(raw_text:str):
        for strategy in [DetokenizerStrategy, FastTextLangDetect, CleanupStrategy]:
            raw_text = strategy().run(raw_text)
        
        raw_text = DetokenizerStrategy().run(raw_text)
    
        raise NotImplementedError("This method should be implemented by the child class")