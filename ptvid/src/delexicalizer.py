import spacy
import random

class Delexicalizer:
    def __init__(self, prob_pos_tag: float, prob_ner_tag: float, spacy_model: str = "pt_core_news_sm") -> None:
        if spacy_model not in spacy.util.get_installed_models():
            spacy.cli.download(spacy_model)
        self.nlp = spacy.load(spacy_model)
        self.prob_pos_tag = prob_pos_tag
        self.prob_ner_tag = prob_ner_tag

    def delexicalize(self, text: str) -> str:
        if self.prob_ner_tag == 0 and self.prob_pos_tag == 0:
            return text

        doc = self.nlp(text)
        list_tokens = []
        for token in doc:
            if token.ent_type > 0 and random.uniform(0, 1) < self.prob_ner_tag:
                list_tokens.append(token.ent_type_)

            elif random.uniform(0, 1) < self.prob_pos_tag:
                list_tokens.append(token.pos_)

            else:
                list_tokens.append(token.text)

        return " ".join(list_tokens)
