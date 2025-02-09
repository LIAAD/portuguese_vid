import re
from pt_vid.data.Tokenizer import Tokenizer
from pt_vid.data.cleaning.Strategy import Strategy

class DetokenizerStrategy(Strategy):
    """Based on the TreebankWordDetokenizer from nltk."""

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r"(\S)\s(\'\')"), r"\1\2"),
        (re.compile(r"(\S)\s(»)"), r"\1\2"),
        (re.compile(r"(\'\')\s([.,:)\]>};%])"), r"\1\2"),
        (re.compile(r"''"), '"'),
    ]

    # Undo padding on parentheses.
    PARENS_BRACKETS = [
        (re.compile(r"([\[\(\{\<])\s"), r"\g<1>"),
        (re.compile(r"\s([\]\)\}\>])"), r"\g<1>"),
        (re.compile(r"([\]\)\}\>])\s([:;,.])"), r"\1\2"),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r"([^'])\s'\s"), r"\1' "),
        (re.compile(r"\s([?!.])"), r"\g<1>"),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r"\1\2\3"),
        (re.compile(r"([#$])\s"), r"\g<1>"),
        (re.compile(r"\s([;%])"), r"\g<1>"),
        (re.compile(r"\s\.\.\.\s"), r"..."),
        (re.compile(r"\s([:,])"), r"\1"),
    ]

    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r"([ (\[{<])\s``"), r"\1``"),
        (re.compile(r"(``)\s"), r"\1"),
        (re.compile(r"(`)\s"), r"\1"),
        (re.compile(r"(«)\s"), r"\1"),
        (re.compile(r"``"), r'"'),
    ]

    PRONOUNS = [
        " -me", 
        " -te", 
        " -se", 
        " -nos", 
        " -vos", 
        " -o", 
        " -a", 
        " -os", 
        " -as", 
        " -lhe", 
        " -lhes", 
        " -lho", 
        " -lha", 
        " -lhos", 
        " -lhas"
    ]

    # Regex to remove space before hyphen if it is connected to a pronoun
    PRONOUNS_REGEX = re.compile(r"(\S)\s(-" + "|".join(PRONOUNS) + r")")

    def _run(text):
        """Duck-typing the abstract *tokenize()*."""
        result = []
        quote_count = 0

        for token in Tokenizer.tokenize(text):
            if token == '"':
                result.append('"' if quote_count % 2 == 0 else '" ')
                quote_count += 1
            else:
                result.append(token + ' ')

        text = " " + "".join(result) + " "

        for regexp, substitution in DetokenizerStrategy.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        text = text.strip()

        for regexp, substitution in DetokenizerStrategy.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        for regexp, substitution in DetokenizerStrategy.PUNCTUATION:
            text = regexp.sub(substitution, text)

        for regexp, substitution in DetokenizerStrategy.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for pronoun in DetokenizerStrategy.PRONOUNS:
            text = text.replace(pronoun, pronoun.strip())

        return text.strip()


    def run(dataset, domain):
        for row in dataset:
            row["text"] = DetokenizerStrategy._run(row["text"])

        return dataset
