# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from structure.corpus import Corpus
from visualization.visualization import Visualization
import stats
import pickle
import utils

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

corpus = Corpus(text_file_path='input/egc/abstracts.txt',
                time_file_path='input/egc/dates.txt',
                language='french',
                max_relative_frequency=0.8,
                min_absolute_frequency=4)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=15)
viz = Visualization(topic_model)
print '\nTopics:'
topic_model.print_topics(num_words=10, display_weights=True)
print '\nDocument 2:', topic_model.corpus.documents[2]
print '\nVector representation of document 2:\n', topic_model.corpus.get_vector_for_document(2)
print '\nTopic distribution for document 2:', topic_model.topic_distribution(2)
viz.plot_topic_distribution(2)
print '\nMost likely topic for document 2:', topic_model.most_likely_topic_for_document(2)
print '\nTopics frequency:', topic_model.topics_frequency()
print '\nTopic 2 frequency:', topic_model.topic_frequency(2)
print '\nTop 10 most likely words for topic 2:', topic_model.get_top_words(2, 10)
viz.plot_word_distribution(2, 20)
utils.save_word_distribution(topic_model.get_top_words(2, 20), 'output/word_distribution.tsv')

# Topic evolution
evolution = []
for i in range(2004, 2016):
    evolution.append((i, topic_model.topic_frequency(2, date=i)))
utils.save_topic_evolution(evolution, 'output/topic_evolution.tsv')

# Associate documents with topics
topic_affiliations = topic_model.documents_per_topic()
document_for_topic2 = topic_affiliations[2]
with open('input/egc/titles0.txt') as f:
    titles = f.read().splitlines()
print '\nDocuments related to topic 2:'
for i in document_for_topic2:
    print titles[i]

# Generate the topic cloud
viz.topic_cloud()
