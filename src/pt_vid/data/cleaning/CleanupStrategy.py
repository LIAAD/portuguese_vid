from pt_vid.data.cleaning.Strategy import Strategy

class CleanupStrategy(Strategy):
    @staticmethod
    def _clean_nan(text):
        pass
    
    @staticmethod
    def _clean_empty_string(text):
        pass
    
    @staticmethod
    def _clean_text(text):
        return clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=False,
            strip_lines=True,
            normalize_whitespace=True,
            no_emoji=True,
            lang="en",
        )
    
    @staticmethod
    def run(text):
        pass