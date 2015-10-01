# coding: utf-8
import matplotlib.pyplot as plt
import codecs
import numpy as np
import seaborn
import json

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Visualization:

    def __init__(self, topic_model):
        self.topic_model = topic_model

    def plot_topic_distribution(self, doc_id, file_path='output/topic_distribution.png'):
        distribution = self.topic_model.topic_distribution_for_document(doc_id)
        data_x = range(0, len(distribution))
        plt.clf()
        plt.xticks(np.arange(0, len(distribution), 1.0))
        plt.bar(data_x, distribution, align='center')
        plt.title('Topic distribution')
        plt.ylabel('probability')
        plt.xlabel('topic')
        plt.savefig(file_path)

    def plot_word_distribution(self, topic_id, nb_words=10, file_path='output/word_distribution.png'):
        data_x = []
        data_y = []
        distribution = self.topic_model.top_words(topic_id, nb_words)
        for weighted_word in distribution:
            data_x.append(weighted_word[0])
            data_y.append(weighted_word[1])
        plt.clf()
        plt.bar(range(len(data_x)), data_y, align='center')
        plt.xticks(range(len(data_x)), data_x, size='small', rotation='vertical')
        plt.title('Word distribution')
        plt.ylabel('probability')
        plt.xlabel('word')
        plt.savefig(file_path)

    def plot_arun_metric(self, min_num_topics=10, max_num_topics=50, iterations=10, file_path='output/arun.png'):
        symmetric_kl_divergence = self.topic_model.arun_metric(min_num_topics, max_num_topics, iterations)
        plt.clf()
        plt.plot(range(min_num_topics, max_num_topics+1), symmetric_kl_divergence)
        plt.title('Arun et al. metric')
        plt.xlabel('number of topics')
        plt.ylabel('symmetric KL divergence')
        plt.savefig(file_path)

    def topic_cloud(self, file_path='output/topic_cloud.json'):
        json_graph = {}
        json_nodes = []
        json_links = []
        for i in range(self.topic_model.nb_topics):
            description = []
            for weighted_word in self.topic_model.top_words(i, 5):
                description.append(weighted_word[1])
            json_nodes.append({'name': 'topic'+str(i),
                               'frequency': self.topic_model.topic_frequency(i),
                               'description': ', '.join(description),
                               'group': i})
        json_graph['nodes'] = json_nodes
        json_graph['links'] = json_links
        with codecs.open(file_path, 'w', encoding='utf-8') as fp:
            json.dump(json_graph, fp, indent=4, separators=(',', ': '))
