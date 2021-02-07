import spacy
from spacy_lefff import LefffLemmatizer
from typing import Iterable, List, Text

class FrenchLemmatizer(object):

    def __init__(self):
        self.nlp = spacy.load('fr')
        self.lemmatizer = LefffLemmatizer()
        self.nlp.add_pipe(self.lemmatizer, name='lefff')

    def lemmatize(self, sentence: Iterable[Text]) -> List[Text]:
        doc = self.nlp(" ".join(sentence))
        return [d.lemma_ for d in doc]

    def lemmatize_s(self, sentence: Text) -> List[Text]:
        doc = self.nlp(sentence)
        return [d.lemma_ for d in doc]
