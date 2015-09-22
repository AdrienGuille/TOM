# coding: utf-8
from abc import ABCMeta, abstractmethod
from gensim import models, matutils
from sklearn.decomposition import NMF
import numpy
import math
import stats

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, corpus):
        self.corpus = corpus  # list of documents (a document is represented by a string)
        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    @abstractmethod
    def infer_topics(self, num_topics=10):
        pass

    def print_topics(self, num_words=10, display_weights=False):
        count = 0
        for topic in self.topic_word_matrix:
            word_list = []
            for weighted_word in topic:
                if display_weights:
                    word_list.append(weighted_word[1]+' ('+str(round(weighted_word[0], 4))+')')
                else:
                    word_list.append(weighted_word[1])
                if len(word_list) == num_words:
                    break
            print 'topic', count, ': ', ' '.join(word_list)
            count += 1

    def get_top_words(self, topic_id, num_words):
        return self.topic_word_matrix[topic_id][:num_words]

    def topic_distribution_for_document(self, doc_id):
        return list(self.document_topic_matrix[doc_id])

    def most_likely_topic_for_document(self, doc_id):
        td = self.topic_distribution_for_document(doc_id)
        return td.index(max(td))

    def topic_distribution_for_word(self, word_id):
        word = self.corpus.get_word_for_id(word_id)
        distribution = []
        for i in range(self.nb_topics):
            for weighted_word in self.topic_word_matrix[i]:
                if weighted_word[1] == word:
                    distribution.append(weighted_word[0])
                    break
        return distribution

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = [0.0] * self.nb_topics
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.get_ids(date)
        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0/len(ids)
        return frequency

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
                              id2word=self.corpus.vocabulary,
                              iterations=10000,
                              num_topics=num_topics)
        self.topic_word_matrix = list(lda.show_topics(num_topics=num_topics,
                                                 num_words=len(self.corpus.vocabulary),
                                                 formatted=False))
        self.document_topic_matrix = numpy.transpose(matutils.corpus2dense(lda[self.corpus.gensim_tfidf],
                                                                           num_topics,
                                                                           self.corpus.size))


class LatentSemanticAnalysis(TopicModel):

    def infer_topics(self, num_topics=10):
        self.nb_topics = num_topics
        lsa = models.LsiModel(corpus=self.corpus.gensim_tfidf,
                              id2word=self.corpus.vocabulary,
                              num_topics=num_topics)
        self.topic_word_matrix = list(lsa.show_topics(num_topics=num_topics,
                                                 num_words=len(self.corpus.vocabulary),
                                                 formatted=False))
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
        for topic_idx, topic in enumerate(nmf.components_):
            word_list = [feature_names[i] for i in topic.argsort()[:-vocabulary_size - 1:-1]]
            weight_list = [topic[i] for i in topic.argsort()[:-vocabulary_size - 1:-1]]
            weighted_word_list = []
            for i in range(0, len(word_list)):
                weighted_word_list.append((weight_list[i], word_list[i]))
            self.topic_word_matrix.append(weighted_word_list)
        for doc in topic_document:
            row = []
            for topic in doc:
                row.append(math.fabs(topic))
            self.document_topic_matrix.append(row)
