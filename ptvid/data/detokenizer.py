import re
from typing import List


class PortugueseDetokenizer:
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

    def detokenize(self, tokens: List[str]) -> str:
        """Duck-typing the abstract *tokenize()*."""
        
        quote_count = 0
        text = ""
        for token in tokens:
            if token == '"':
                if quote_count % 2 == 0:
                    text += '"'
                else:
                    text = text[:-1]
                    text += '" '
                quote_count += 1
            else:
                text += f'{token} '

        # Add extra space to make things easier
        text = " " + text + " "

        # Reverse the regexes applied for ending quotes.
        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        # Undo the space padding.
        text = text.strip()

        text = regexp.sub(substitution, text)

        # Reverse the padding regexes applied for parenthesis/brackets.
        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for punctuations.
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Reverse the regexes applied for starting quotes.
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for pronoun in self.PRONOUNS:
            text = text.replace(pronoun, pronoun.strip())

        quote_count = 0


        return text.strip()

