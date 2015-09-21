# coding: utf-8
import codecs
from gensim import matutils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:

    def __init__(self,
                 full_content_file_path,
                 author_file_path,
                 short_content_file_path=None,
                 time_file_path=None,
                 language=None,
                 max_relative_frequency=0.95,
                 min_absolute_frequency=2):
        with codecs.open(full_content_file_path, 'r', encoding='utf-8') as f:
            self.documents = f.read().splitlines()
        with codecs.open(author_file_path, 'r', encoding='utf-8') as f:
            tmp_author_list = f.read().splitlines()
        self.authors = []
        for author_list in tmp_author_list:
            self.authors.append(author_list.split(', '))
        self.size = len(self.documents)
        self.dates = None
        self.time_index = None
        self.titles = None
        if short_content_file_path is not None:
            with codecs.open(short_content_file_path, 'r', encoding='utf-8') as f:
                self.titles = f.read().splitlines()
        if time_file_path is not None:
            with codecs.open(time_file_path, 'r', encoding='utf-8') as f:
                self.dates = f.read().splitlines()
            self.time_index = {}
            for i in range(0, self.size):
                date = self.dates[i]
                docs = self.time_index.get(date)
                if docs is None:
                    docs = [i]
                else:
                    docs.append(i)
                self.time_index[date] = docs
        stop_words = []
        if language is not None:
            stop_words = stopwords.words(language)
        self.vectorizer = TfidfVectorizer(max_df=max_relative_frequency,
                                          min_df=min_absolute_frequency,
                                          max_features=2000,
                                          stop_words=stop_words)
        self.sklearn_tfidf = self.vectorizer.fit_transform(self.documents)
        self.gensim_tfidf = matutils.Sparse2Corpus(self.sklearn_tfidf, documents_columns=False)
        vocab = self.vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def documents_by_author(self, author):
        ids = []
        for i in range(self.size):
            if self.is_author(author, i):
                ids.append(i)
        return ids

    def is_author(self, author, doc_id):
        return author in self.authors[doc_id]

    def get_ids(self, date):
        if self.time_index is not None:
            return self.time_index.get(str(date))
        else:
            raise ValueError('No temporal information available for this corpus')

    def get_vector_for_document(self, doc_id):
        return self.sklearn_tfidf[doc_id]

    def get_word_for_id(self, word_id):
        return self.vocabulary.get(word_id)

    def collaboration_network(self, doc_ids=None):
        nx_graph = nx.Graph(name='')
        json_graph = {}
        json_nodes = []
        node_array = []
        json_links = []
        if doc_ids is None:
            doc_ids = range(self.size)
        distinct_authors = set()
        for doc_id in doc_ids:
            collaborators = self.authors[doc_id]
            for collaborator in collaborators:
                distinct_authors.add(collaborator)
        node_array = list(distinct_authors)
        nx_graph.add_nodes_from(node_array)
        for doc_id in doc_ids:
            collaborators = self.authors[doc_id]
            for j in range(0, len(collaborators)):
                for i in range(j+1, len(collaborators)):
                    json_links.append({'source': node_array.index(collaborators[i]), 'target': node_array.index(collaborators[j]), 'value':1})
                    nx_graph.add_edge(collaborators[i], collaborators[j])
        node_array = []
        group = 0
        page_rank = nx.pagerank(nx_graph, alpha=0.85, max_iter=300)
        for doc_id in doc_ids:
            collaborators = self.authors[doc_id]
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
