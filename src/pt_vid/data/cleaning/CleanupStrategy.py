import numpy as np
from cleantext import clean
from pt_vid.data.cleaning.Strategy import Strategy
class CleanupStrategy(Strategy):
    @staticmethod
    def _clean_nan(text):
        return "" if text is np.nan else text
     
    @staticmethod
    def _clean_text(text, **kwargs):
        return clean(
            text,
            fix_unicode=kwargs.get("fix_unicode", True),
            to_ascii=kwargs.get("to_ascii", False),
            lower=kwargs.get("lower", False),
            no_line_breaks=kwargs.get("no_line_breaks", True),
            no_urls=kwargs.get("no_urls", True),
            no_emails=kwargs.get("no_emails", True),
            no_phone_numbers=kwargs.get("no_phone_numbers", True),
            no_numbers=kwargs.get("no_numbers", True),
            no_digits=kwargs.get("no_digits", True),
            no_currency_symbols=kwargs.get("no_currency_symbols", True),
            no_punct=kwargs.get("no_punct", False),
            strip_lines=kwargs.get("strip_lines", True),
            normalize_whitespace=kwargs.get("normalize_whitespace", True),
            no_emoji=kwargs.get("no_emoji", True),
            lang=kwargs.get("lang", "en"),
        )
    
    @staticmethod
    def _run(text):
        text = CleanupStrategy._clean_nan(text)
        text = CleanupStrategy._clean_text(text)

        return text
    
    @staticmethod
    def run(dataset):
        dataset['text'] = dataset['text'].apply(CleanupStrategy._run)

        # Drop rows with empty text
        dataset = dataset[dataset['text'] != '']

        return dataset