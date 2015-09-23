# coding: utf-8
import codecs
import json

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def save_word_distribution(distribution, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('word\tweight\n')
        for weighted_word in distribution:
            f.write(weighted_word[0]+'\t'+str(weighted_word[1])+'\n')


def save_topic_distribution(distribution, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('topic\tweight\n')
        for i in range(len(distribution)):
            f.write('topic '+str(i)+'\t'+str(distribution[i])+'\n')


def save_topic_evolution(evolution, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as f:
        f.write('date\tfrequency\n')
        for date, frequency in evolution:
            f.write(str(date)+'\t'+str(frequency)+'\n')


def save_topic_cloud(topic_model, file_path):
    json_graph = {}
    json_nodes = []
    json_links = []
    for i in range(topic_model.nb_topics):
        description = []
        for weighted_word in topic_model.top_words(i, 5):
            description.append(weighted_word[0])
        json_nodes.append({'name': i,
                           'frequency': topic_model.topic_frequency(i),
                           'description': ', '.join(description),
                           'group': i})
    json_graph['nodes'] = json_nodes
    json_graph['links'] = json_links
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_graph, fp, indent=4, separators=(',', ': '))


def save_author_network(json_graph, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_graph, fp, indent=4, separators=(',', ': '))
