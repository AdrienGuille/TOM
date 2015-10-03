# coding: utf-8
from gensim import matutils
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import itertools
import pandas
from networkx.readwrite import json_graph
from scipy import spatial
import random
__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:

    def __init__(self,
                 source_file_path,
                 language=None,
                 vectorization='tfidf',
                 max_relative_frequency=0.95,
                 min_absolute_frequency=2,
                 preprocessor=None,
                 sample=None):

        self.__source_file_path = source_file_path
        self.__language = language
        self.__vectorization = vectorization
        self.__max_relative_frequency = max_relative_frequency
        self.__min_absolute_frequency = min_absolute_frequency
        self.__preprocessor = preprocessor

        self.data_frame = pandas.read_csv(source_file_path, sep='\t', encoding='utf-8')
        if sample:
            self.data_frame = self.data_frame.sample(frac=random.random())
        self.size = self.data_frame.count(0)[0]
        if preprocessor is not None:
            for i in self.data_frame.index.tolist():
                full_content = self.data_frame.iloc[i]['full_content']
                lemmatized_content = []
                for word in wordpunct_tokenize(full_content.lower()):
                    lemmatized_content.append(preprocessor.get_lemma(word))
                self.data_frame.loc[i, 'full_content'] = ' '.join(lemmatized_content)
        stop_words = []
        if language is not None:
            stop_words = stopwords.words(language)
        if vectorization == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_df=max_relative_frequency,
                                              min_df=min_absolute_frequency,
                                              max_features=2000,
                                              stop_words=stop_words)
        elif vectorization == 'tf':
            self.vectorizer = CountVectorizer(max_df=max_relative_frequency,
                                              min_df=min_absolute_frequency,
                                              max_features=2000,
                                              stop_words=stop_words)
        else:
            raise ValueError('Unknown vectorization type: %s' % vectorization)
        self.sklearn_vector_space = self.vectorizer.fit_transform(self.data_frame['full_content'].tolist())
        self.gensim_vector_space = matutils.Sparse2Corpus(self.sklearn_vector_space, documents_columns=False)
        vocab = self.vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def full_content(self, doc_id):
        return self.data_frame.iloc[doc_id]['full_content']

    def short_content(self, doc_id):
        return self.data_frame.iloc[doc_id]['short_content']

    def date(self, doc_id):
        return self.data_frame.iloc[doc_id]['date']

    def authors(self, doc_id):
        return unicode(self.data_frame.iloc[doc_id]['authors']).split(', ')

    def affiliations(self, doc_id):
        return unicode(self.data_frame.iloc[doc_id]['affiliations']).split(', ')

    def documents_by_author(self, author):
        ids = []
        for i in range(self.size):
            if self.is_author(author, i):
                ids.append(i)
        return ids

    def is_author(self, author, doc_id):
        return author in self.authors(doc_id)

    def docs_for_word(self, word_id):
        ids = []
        for i in range(self.size):
            vector = self.vector_for_document(i)
            if vector[word_id] > 0:
                ids.append(i)
        return ids

    def doc_ids(self, date):
        return self.data_frame[self.data_frame['date'] == date].index.tolist()

    def vector_for_document(self, doc_id):
        vector = self.sklearn_vector_space[doc_id]
        cx = vector.tocoo()
        weights = [0.0] * len(self.vocabulary)
        for row, word_id, weight in itertools.izip(cx.row, cx.col, cx.data):
            weights[word_id] = weight
        return weights

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def similar_documents(self, doc_id, num_docs):
        doc_weights = self.vector_for_document(doc_id)
        similarities = []
        for a_doc_id in range(self.size):
            if a_doc_id != doc_id:
                similarity = 1.0 - spatial.distance.cosine(doc_weights, self.vector_for_document(a_doc_id))
                similarities.append((a_doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]

    def collaboration_network(self, doc_ids=None):
        nx_graph = nx.Graph(name='')
        for doc_id in doc_ids:
            authors = self.authors(doc_id)
            for author in authors:
                nx_graph.add_node(author)
            for i in range(0, len(authors)):
                for j in range(i+1, len(authors)):
                    nx_graph.add_edge(authors[i], authors[j])
        bb = nx.betweenness_centrality(nx_graph)
        nx.set_node_attributes(nx_graph, 'betweenness', bb)
        return json_graph.node_link_data(nx_graph)
