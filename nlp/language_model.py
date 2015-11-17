# coding: utf-8
from nltk.tokenize import WordPunctTokenizer
from structure.corpus import Corpus
import numpy as np

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class LanguageModel:

    def __init__(self, corpus):
        self.corpus = corpus
        self.word_context_matrix = None

    def compute_word_context_matrix(self, window=5):
        tokenizer = WordPunctTokenizer()
        self.word_context_matrix = np.zeros((len(self.corpus.vocabulary), len(self.corpus.vocabulary)))
        for doc_id in range(self.corpus.size):
            print doc_id
            document = self.corpus.full_content(doc_id)
            terms = tokenizer.tokenize(document)
            nb_terms = len(terms)
            for i in range(nb_terms):
                row_index = self.corpus.id_for_word(terms[i])
                if row_index != -1:
                    start = i - window
                    if start < 0:
                        start = 0
                    end = i + window
                    if end >= nb_terms:
                        end = nb_terms-1
                    context0 = terms[start:i]
                    context1 = terms[i+1:end+1]
                    context0.extend(context1)
                    for term in context0:
                        column_index = self.corpus.id_for_word(term)
                        if column_index != -1:
                            self.word_context_matrix[row_index][column_index] += 1

if __name__ == '__main__':
    corpus = Corpus(source_file_path='../input/egc_lemmatized.csv',
                    language='french',
                    vectorization='tfidf',
                    max_relative_frequency=0.8,
                    min_absolute_frequency=4,
                    preprocessor=None)
    model = LanguageModel(corpus)
    model.compute_word_context_matrix(5)
