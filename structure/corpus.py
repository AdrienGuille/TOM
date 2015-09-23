# coding: utf-8
from gensim import matutils
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import itertools
import pandas

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:

    def __init__(self,
                 source_file_path,
                 language=None,
                 max_relative_frequency=0.95,
                 min_absolute_frequency=2,
                 lemmatizer=None):
        self.data_frame = pandas.read_csv(source_file_path, sep='\t', index_col=0, encoding='utf-8')
        self.size = self.data_frame.count(0)[0]
        if lemmatizer is not None:
            for i in self.data_frame.index.tolist():
                full_content = self.data_frame.iloc[i]['full_content']
                lemmatized_content = []
                for word in wordpunct_tokenize(full_content.lower()):
                    lemmatized_content.append(lemmatizer.get_lemma(word))
                self.data_frame.loc[i, 'full_content'] = ' '.join(lemmatized_content)
        stop_words = []
        if language is not None:
            stop_words = stopwords.words(language)
        self.vectorizer = TfidfVectorizer(max_df=max_relative_frequency,
                                          min_df=min_absolute_frequency,
                                          max_features=2000,
                                          stop_words=stop_words)
        self.sklearn_tfidf = self.vectorizer.fit_transform(self.data_frame['full_content'].tolist())
        self.gensim_tfidf = matutils.Sparse2Corpus(self.sklearn_tfidf, documents_columns=False)
        vocab = self.vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def full_content(self, doc_id):
        return self.data_frame.iloc[doc_id]['full_content']

    def short_content(self, doc_id):
        return self.data_frame.iloc[doc_id]['short_content']

    def date(self, doc_id):
        return self.data_frame.iloc[doc_id]['date']

    def authors(self, doc_id):
        return self.data_frame.iloc[doc_id]['authors']

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
        vector = self.sklearn_tfidf[doc_id]
        cx = vector.tocoo()
        word_list = [0.0] * len(self.vocabulary)
        for row, word_id, weight in itertools.izip(cx.row, cx.col, cx.data):
            word_list[word_id] = weight
        return word_list

    def id_for_word(self, word):
        return 0

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def collaboration_network(self, doc_ids=None):
        nx_graph = nx.Graph(name='')
        json_graph = {}
        json_nodes = []
        json_links = []
        if doc_ids is None:
            doc_ids = range(self.size)
        distinct_authors = set()
        for doc_id in doc_ids:
            collaborators = self.authors(doc_id)
            for collaborator in collaborators:
                distinct_authors.add(collaborator)
        node_array = list(distinct_authors)
        nx_graph.add_nodes_from(node_array)
        for doc_id in doc_ids:
            collaborators = self.authors(doc_id)
            for j in range(0, len(collaborators)):
                for i in range(j+1, len(collaborators)):
                    json_links.append({'source': node_array.index(collaborators[i]), 'target': node_array.index(collaborators[j]), 'value':1})
                    nx_graph.add_edge(collaborators[i], collaborators[j])
        node_array = []
        group = 0
        page_rank = nx.pagerank(nx_graph, alpha=0.85, max_iter=300)
        for doc_id in doc_ids:
            collaborators = self.authors(doc_id)
            for author in collaborators:
                if author not in node_array:
                    node_array.append(author)
                    json_nodes.append({'name': author,
                                       'weight': page_rank.get(author)*10000,
                                       'group': -1})
            group += 1
        json_graph['nodes'] = json_nodes
        json_graph['links'] = json_links
        return json_graph
