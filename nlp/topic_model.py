# coding: utf-8
from abc import ABCMeta, abstractmethod
from gensim import matutils, models
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class TopicModel(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 raw_data=None,
                 language=None,
                 max_relative_frequency=0.95,
                 min_absolute_frequency=2,
                 vectorized_data=None):
        if vectorized_data is None:
            stop_words = []
            if language is not None:
                stop_words = stopwords.words(language)
            self.vectorizer = TfidfVectorizer(max_df=max_relative_frequency,
                                              min_df=min_absolute_frequency,
                                              max_features=2000,
                                              stop_words=stop_words)
            self.sklearn_corpus = self.vectorizer.fit_transform(raw_data)
            self.gensim_corpus = matutils.Sparse2Corpus(self.sklearn_corpus, documents_columns=False)
            vocab = self.vectorizer.get_feature_names()
            self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])
        else:
            self.vectorizer = vectorized_data.vectorizer
            self.sklearn_corpus = vectorized_data.sklearn_corpus
            self.gensim_corpus = vectorized_data.gensim_corpus
            self.vocabulary = vectorized_data.vocabulary
        self.topic_document_matrix = None
        self.word_topic_matrix = None

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    @abstractmethod
    def infer_topics(self, num_topics=10):
        pass

    def print_topics(self, num_words=10, display_weights=False):
        count = 0
        for topic in self.word_topic_matrix:
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

    def print_topics_for_document(self, doc_id):
        count = 0
        for doc in self.topic_document_matrix:
            if count == doc_id:
                print doc
                break
            count += 1


class LatentDirichletAllocation(TopicModel):

    def infer_topics(self, num_topics=10):
        lda = models.LdaModel(corpus=self.gensim_corpus,
                              id2word=self.vocabulary,
                              iterations=3000,
                              num_topics=num_topics)
        self.word_topic_matrix = lda.show_topics(num_topics=num_topics,
                                                 num_words=len(self.vocabulary),
                                                 formatted=False)
        self.topic_document_matrix = lda[self.gensim_corpus]


class LatentSemanticAnalysis(TopicModel):

    def infer_topics(self, num_topics=10):
        lsa = models.LsiModel(corpus=self.gensim_corpus,
                              id2word=self.vocabulary,
                              num_topics=num_topics)
        self.word_topic_matrix = lsa.show_topics(num_topics=num_topics,
                                                 num_words=len(self.vocabulary),
                                                 formatted=False)
        self.topic_document_matrix = lsa[self.gensim_corpus]


class NonNegativeMatrixFactorization(TopicModel):

    def infer_topics(self, num_topics=10):
        nmf = NMF(n_components=num_topics, random_state=1)
        topic_document = nmf.fit_transform(self.sklearn_corpus)
        feature_names = self.vectorizer.get_feature_names()
        self.word_topic_matrix = []
        self.topic_document_matrix = []
        vocabulary_size = len(self.vocabulary)
        for topic_idx, topic in enumerate(nmf.components_):
            word_list = [feature_names[i] for i in topic.argsort()[:-vocabulary_size - 1:-1]]
            weight_list = [topic[i] for i in topic.argsort()[:-vocabulary_size - 1:-1]]
            weighted_word_list = []
            for i in range(0, len(word_list)):
                weighted_word_list.append((weight_list[i], word_list[i]))
            self.word_topic_matrix.append(weighted_word_list)
        for doc in topic_document:
            row = []
            i = 0
            for topic in doc:
                row.append((i, topic))
                i += 1
            self.topic_document_matrix.append(row)
