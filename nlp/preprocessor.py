# coding: utf-8
import codecs
from abc import ABCMeta, abstractmethod
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer as SnowballEnglishStemmer
from nltk.corpus.reader import wordnet
import nltk
import subprocess

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def to_wordnet_tag(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''


class PreProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process_sentence(self, sentence):
        pass


class FrenchLemmatizer(PreProcessor):

    def __init__(self):
        self.script_path = 'nlp/melt.sh'

    def process_sentence(self, sentence):
        ps = subprocess.Popen(('echo', sentence), stdout=subprocess.PIPE)
        output = subprocess.check_output(('MElt', '-L'), stdin=ps.stdout)
        ps.wait()
        stemmed_sentence = []
        for annotated_token in output.split(' '):
            tag = annotated_token.split('/')
            stemmed_sentence.append(tag[2])
        print sentence, stemmed_sentence
        return ' '.join(stemmed_sentence)


class EnglishStemmer(PreProcessor):

    def __init__(self):
        self.stemmer = SnowballEnglishStemmer()

    def process_sentence(self, sentence):
        stemmed_sentence = []
        for token in wordpunct_tokenize(sentence):
            if len(token) > 1:
                stemmed_sentence.append(self.stemmer.stem(token))
        return ' '.join(stemmed_sentence)


class EnglishLemmatizer(PreProcessor):

    def __init__(self, skip_token_without_pos):
        self.lemmatizer = WordNetLemmatizer()
        self.skip = skip_token_without_pos

    def process_sentence(self, sentence):
        tokenized_sentence = nltk.word_tokenize(sentence)
        lemmatized_sentence = []
        for token, treebank_pos_tag in nltk.pos_tag(tokenized_sentence):
            wordnet_pos_tag = to_wordnet_tag(treebank_pos_tag)
            if wordnet_pos_tag != '':
                lemmatized_sentence.append(self.lemmatizer.lemmatize(token, wordnet_pos_tag).lower())
            elif not self.skip and len(token) > 1:
                lemmatized_sentence.append(token.lower())
        return ' '.join(lemmatized_sentence)
