# coding: utf-8
import platform

import numpy as np
from tom_lib.nlp.preprocessor import EnglishLemmatizer
from nltk.tokenize import WordPunctTokenizer
from scipy import spatial
from sklearn import decomposition

from tom_lib.structure.corpus import Corpus

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class SemanticModel:

    def __init__(self, corpus, window=5):
        self.corpus = corpus
        tokenizer = WordPunctTokenizer()
        self.word_context_matrix = np.zeros((len(self.corpus.vocabulary), len(self.corpus.vocabulary)))
        for doc_id in range(self.corpus.size):
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

    def ppmi_transform(self, laplace_smoothing=0):
        n_r = len(self.word_context_matrix[:, 0])
        n_c = len(self.word_context_matrix[0, :])
        ppmi_matrix = np.zeros((n_r, n_c))
        total_f_ij = 0
        for i in range(n_r):
            for j in range(n_c):
                total_f_ij += self.word_context_matrix[i, j]+laplace_smoothing
        for i in range(n_r):
            for j in range(n_c):
                p_ij = (self.word_context_matrix[i, j]+laplace_smoothing)/total_f_ij
                p_i = (np.sum(self.word_context_matrix[i, :])+n_c*laplace_smoothing)/total_f_ij
                p_j = (np.sum(self.word_context_matrix[:, j])+n_r*laplace_smoothing)/total_f_ij
                if p_ij/(p_i*p_j) > 0:
                    pmi_ij = np.log10(p_ij/(p_i*p_j))
                    if pmi_ij < 0:
                        pmi_ij = 0
                else:
                    pmi_ij = 0
                ppmi_matrix[i, j] = pmi_ij
        self.word_context_matrix = ppmi_matrix

    def svd_smoothing(self, dimension=None):
        if dimension is None:
            dimension = int(len(self.word_context_matrix[0, :])/10)
        svd = decomposition.TruncatedSVD(n_components=dimension)
        self.word_context_matrix = svd.fit_transform(self.word_context_matrix)

    def most_similar_words(self, word_vector, nb_words=3):
        similarity = []
        for i in range(len(self.word_context_matrix[:, 0])):
            similarity.append(spatial.distance.cosine(word_vector, self.word_context_matrix[i, :]))
        similar_word_ids = np.argsort(np.array(similarity)).tolist()
        return similar_word_ids[:nb_words]

if __name__ == '__main__':
    input_file_path = ''
    output_file_path = ''
    if platform.system() == 'Darwin':
        input_file_path = '/Users/adrien/data/HP/HP1.csv'
        output_file_path = '/Users/adrien/data/HP/HP1_lemmatized.csv'
    elif platform.system() == 'Linux':
        input_file_path = '/home/adrien/datasets/HP/HP1.csv'
        output_file_path = '/home/adrien/datasets/HP/HP1_lemmatized.csv'
    print('Loading corpus...')
    corpus = Corpus(source_file_path=input_file_path,
                    vectorization='tf',
                    max_relative_frequency=0.75,
                    min_absolute_frequency=4,
                    preprocessor=EnglishLemmatizer())
    print(' - corpus size:', corpus.size)
    print(' - vocabulary size:', len(corpus.vocabulary))
    corpus.export(output_file_path)

    print('Computing semantic model...')
    print(' - calculating raw frequencies...')
    model = SemanticModel(corpus, window=7)

    print(' - transforming raw frequencies (Positive Pointwise Mutual Information)...')
    model.ppmi_transform(laplace_smoothing=2)

    print(' - smoothing model (SVD)...')
    model.svd_smoothing(dimension=300)

    while True:
        choice = input('type l to lookup a word\'s id, s to find synonyms of a word')
        if choice == 'l':
            a_word = input('Word: ')
            print('This word\'s id is: ', corpus.id_for_word(a_word))
        if choice == 's':
            a_word_id = input('Word id: ')
            print('This id corresponds to: ', corpus.word_for_id(a_word_id))
            for similar_word_id in model.most_similar_words(model.word_context_matrix[a_word_id, :], 5):
                print(corpus.word_for_id(similar_word_id))
        elif choice == '-':
            word_a = input('word (a) id: ')
            word_b = input('word (b) id: ')
            word_c = np.substract(model.word_context_matrix[word_a, :], model.word_context_matrix[word_b, :])
            for similar_word_id in model.most_similar_words(word_c, 5):
                print(corpus.word_for_id(similar_word_id))
            else:
                word_a = input('word (a) id: ')
                word_b = input('word (b) id: ')
                word_c = np.add(model.word_context_matrix[word_a, :], model.word_context_matrix[word_b, :])
                for similar_word_id in model.most_similar_words(word_c, 5):
                    print(corpus.word_for_id(similar_word_id))
