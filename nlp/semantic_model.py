# coding: utf-8
from nltk.tokenize import WordPunctTokenizer
from nlp.preprocessor import FrenchLemmatizer
from structure.corpus import Corpus
import numpy as np
from sklearn import decomposition
from scipy import spatial

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
                pmi_ij = np.log(p_ij/(p_i*p_j))
                if pmi_ij < 0:
                    pmi_ij = 0
                ppmi_matrix[i, j] = pmi_ij
        self.word_context_matrix = ppmi_matrix

    def svd_smoothing(self, dimension=None):
        if dimension is None:
            dimension = int(len(self.word_context_matrix[0, :])/10)
        svd = decomposition.TruncatedSVD(n_components=dimension)
        self.word_context_matrix = svd.fit_transform(self.word_context_matrix)

    def most_similar_words(self, word_id, nb_words=3):
        similarity = []
        input_vector = self.word_context_matrix[word_id, :]
        for i in range(len(self.word_context_matrix[:, 0])):
            if i != word_id:
                similarity.append(spatial.distance.cosine(input_vector, self.word_context_matrix[i, :]))
            else:
                similarity.append(0)
        similar_word_id = np.argsort(np.array(similarity)).tolist()
        return similar_word_id[:nb_words]

if __name__ == '__main__':
    print 'Loading corpus...'
    corpus = Corpus(source_file_path='../input/mockup.csv',
                    language='french',
                    vectorization='tfidf',
                    max_relative_frequency=0.8,
                    min_absolute_frequency=4,
                    preprocessor=FrenchLemmatizer())
    print ' - corpus size:', corpus.size
    print ' - vocabulary size:', len(corpus.vocabulary)
    corpus.export('../input/elysee_lemmatized_50.csv')

    print 'Computing semantic model...'
    print ' - calculating raw frequencies...'
    model = SemanticModel(corpus, window=10)

    print ' - transforming raw frequencies (Positive Pointwise Mutual Information)...'
    model.ppmi_transform(laplace_smoothing=1)

    print ' - smoothing model (SVD)...'
    model.svd_smoothing(dimension=100)

    while True:
        word = str(input('Word: '))
        word_id = corpus.id_for_word(word)
        print 'This id corresponds to:', corpus.word_for_id(word_id)
        print 'Most similar words:'
        for similar_id in model.most_similar_words(word_id):
            print corpus.word_for_id(similar_id)
