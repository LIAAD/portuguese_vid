import nltk

if not nltk.download('punkt'):
    nltk.download('punkt')

if not nltk.download('punkt_tab'):
    nltk.download('punkt_tab')

class Tokenizer:
    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text, language="portuguese")