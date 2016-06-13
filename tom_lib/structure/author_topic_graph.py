# coding: utf-8
import networkx as nx
from networkx.readwrite import json_graph

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class AuthorTopicGraph:

    def __init__(self, topic_model):
        self.topic_model = topic_model  # a TopicModel object
        self.graph = nx.Graph(name='Author-topic graph')
        topic_associations = self.topic_model.documents_per_topic()
        for topic_id in range(self.topic_model.nb_topics):
            self.graph.add_node(topic_id, node_class='topic')
            doc_ids = topic_associations[topic_id]
            authors = []
            for doc_id in doc_ids:
                authors.extend(self.topic_model.corpus.author(doc_id))
            authors = set(authors)
            for author in authors:
                self.graph.add_node(author, node_class='author')
                self.graph.add_edge(author, topic_id)
        self.json_graph = json_graph.node_link_data(self.graph)
