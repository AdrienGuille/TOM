# coding: utf-8
import codecs
from abc import ABCMeta, abstractmethod
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class PreProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process_sentence(self, word):
        pass


class FrenchLemmatizer(PreProcessor):

    def __init__(self):
        input_file = codecs.open('nlp/lexicons/lefff-3.4.mlex', 'r', encoding='utf-8')
        self.table = {}
        for line in input_file:
            line = line.lower()
            entry = line.split('\t')
            self.table[entry[0]] = entry[2]

    def process_sentence(self, word):
        lemma = self.table.get(word)
        if lemma is None:
            lemma = word
        return lemma


class EnglishStemmer(PreProcessor):

    def __init__(self):
        self.wordnet = WordNetLemmatizer()

    def process_sentence(self, sentence):
        output = []
        for word in wordpunct_tokenize(sentence):
            output.append(self.wordnet.lemmatize(word))
        return ' '.join(output)
