import spacy
import random


class Delexicalizer:
    def __init__(self, prob_pos_tag, prob_ner_tag, spacy_model="pt_core_news_sm") -> None:

        if not spacy_model in spacy.util.get_installed_models():
            spacy.cli.download(spacy_model)

        self.nlp = spacy.load(spacy_model)

        if prob_pos_tag < 0 or prob_pos_tag > 1:
            raise ValueError("prob_pos_tag must be between 0 and 1")

        if prob_ner_tag < 0 or prob_ner_tag > 1:
            raise ValueError("prob_ner_tag must be between 0 and 1")

        self.prob_pos_tag = prob_pos_tag
        self.prob_ner_tag = prob_ner_tag

    def delexicalize(self, text):
        doc = self.nlp(text)

        list_tokens = []

        for token in doc:

            if token.ent_type > 0 and random.uniform(0, 1) < self.prob_ner_tag:
                list_tokens.append(token.ent_type_)

            elif random.uniform(0, 1) < self.prob_pos_tag:
                list_tokens.append(token.pos_)

            else:
                list_tokens.append(token.text)

        return ' '.join(list_tokens)