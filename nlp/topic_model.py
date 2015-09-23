# coding: utf-8
from abc import ABCMeta, abstractmethod
from gensim import models, matutils
from sklearn.decomposition import NMF
import numpy
import math
from scipy.sparse import coo_matrix
import stats
from collections import defaultdict
import itertools

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus  # a Corpus object
        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None

    @abstractmethod
    def infer_topics(self, num_topics=10):
        pass

    def print_topics(self, num_words=10):
        for topic_id in range(self.nb_topics):
            word_list = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
            print 'topic', topic_id, ': ', ' '.join(word_list)

    def estimate_optimal_k(self):
        U, s, V = numpy.linalg.svd(self.topic_word_matrix, full_matrices=True)
        return ""

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.izip(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1])
        weighted_words.reverse()
        return weighted_words[:num_words]

    def topic_distribution_for_document(self, doc_id):
        return list(self.document_topic_matrix[doc_id])

    def most_likely_topic_for_document(self, doc_id):
        td = self.topic_distribution_for_document(doc_id)
        return td.index(max(td))

    def topic_distribution_for_word(self, word_id):
        vector = self.topic_word_matrix[:, word_id]
        cx = vector.tocoo()
        distribution = []
        for row, topic_id, weight in itertools.izip(cx.row, cx.col, cx.data):
            distribution.append(weight)
        return distribution

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = [0.0] * self.nb_topics
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date)
        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0/len(ids)
        return frequency

    def top_documents(self, topic_id, num_docs):
        vector = self.document_topic_matrix[:, topic_id]
        return vector.argsort()[:num_docs+1:-1]

    def documents_per_topic(self):
        affiliations = {}
        for i in range(self.corpus.size):
            topic_id = self.most_likely_topic_for_document(i)
            if affiliations.get(topic_id):
                documents = affiliations[topic_id]
                documents.append(i)
                affiliations[topic_id] = documents
            else:
                documents = [topic_id]
                affiliations[topic_id] = documents
        return affiliations


class LatentDirichletAllocation(TopicModel):

    def infer_topics(self, num_topics=10):
        self.nb_topics = num_topics
        lda = models.LdaModel(corpus=self.corpus.gensim_tfidf,
                              iterations=10000,
                              num_topics=num_topics)
        tmp_topic_word_matrix = list(lda.show_topics(num_topics=num_topics,
                                                     num_words=len(self.corpus.vocabulary),
                                                     formatted=False))
        row = []
        col = []
        data = []
        for topic_id in range(self.nb_topics):
            topic_description = tmp_topic_word_matrix[topic_id]
            for probability, word_id in topic_description:
                row.append(topic_id)
                col.append(word_id)
                data.append(probability)
        self.topic_word_matrix = coo_matrix((data, (row, col)), shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        self.document_topic_matrix = numpy.transpose(matutils.corpus2dense(lda[self.corpus.gensim_tfidf],
                                                                           num_topics,
                                                                           self.corpus.size))


class LatentSemanticAnalysis(TopicModel):

    def infer_topics(self, num_topics=10):
        self.nb_topics = num_topics
        lsa = models.LsiModel(corpus=self.corpus.gensim_tfidf,
                              id2word=self.corpus.vocabulary,
                              num_topics=num_topics)
        tmp_topic_word_matrix = list(lsa.show_topics(num_topics=num_topics,
                                                     num_words=len(self.corpus.vocabulary),
                                                     formatted=False))
        row = []
        col = []
        data = []
        for topic_id in range(self.nb_topics):
            topic_description = tmp_topic_word_matrix[topic_id]
            for weight, word_id in topic_description:
                row.append(topic_id)
                col.append(word_id)
                data.append(weight)
        self.topic_word_matrix = coo_matrix((data, (row, col)), shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        self.document_topic_matrix = numpy.transpose(matutils.corpus2dense(lsa[self.corpus.gensim_tfidf],
                                                                           num_topics,
                                                                           self.corpus.size))


class NonNegativeMatrixFactorization(TopicModel):

    def infer_topics(self, num_topics=10):
        self.nb_topics = num_topics
        nmf = NMF(n_components=num_topics, random_state=1)
        topic_document = nmf.fit_transform(self.corpus.sklearn_tfidf)
        feature_names = self.corpus.vectorizer.get_feature_names()
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(nmf.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)), shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        for doc in topic_document:
            row = []
            for topic in doc:
                row.append(math.fabs(topic))
            self.document_topic_matrix.append(row)
