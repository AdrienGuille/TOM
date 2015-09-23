# coding: utf-8
import codecs
from abc import ABCMeta, abstractmethod
from nltk.stem import WordNetLemmatizer

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Lemmatizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_lemma(self, word):
        pass


class FrenchLefff(Lemmatizer):

    def __init__(self):
        input_file = codecs.open('nlp/lexicons/lefff-3.4.mlex', 'r', encoding='utf-8')
        self.table = {}
        for line in input_file:
            line = line.lower()
            entry = line.split('\t')
            self.table[entry[0]] = entry[2]

    def get_lemma(self, word):
        lemma = self.table.get(word)
        if lemma is None:
            lemma = word
        return lemma


class EnglishWordNet(Lemmatizer):

    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def get_lemma(self, word):
        return self.wordnet_lemmatizer.lemmatize(word)
