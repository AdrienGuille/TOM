# coding: utf-8
import tom_lib.utils as ut
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization
from tom_lib.structure.corpus import Corpus
from tom_lib.visualization.visualization import Visualization

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"

# Load and prepare a corpus
print('Load documents from CSV')
corpus = Corpus(source_file_path='input/egc_lemmatized.csv',
                language='french',  # language for stop words
                vectorization='tfidf',  # 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
                max_relative_frequency=0.8,  # ignore words which relative frequency is > than max_relative_frequency
                min_absolute_frequency=4)  # ignore words which absolute frequency is < than min_absolute_frequency
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))
print('Vector representation of document 0:\n', corpus.vector_for_document(0))

# Instantiate a topic model
topic_model = NonNegativeMatrixFactorization(corpus)

# Estimate the optimal number of topics
print('Estimating the number of topics...')
viz = Visualization(topic_model)
viz.plot_greene_metric(min_num_topics=10,
                       max_num_topics=11,
                       tao=10, step=1,
                       top_n_words=10)
viz.plot_arun_metric(min_num_topics=5,
                     max_num_topics=30,
                     iterations=10)
viz.plot_brunet_metric(min_num_topics=5,
                       max_num_topics=30,
                       iterations=10)

# Infer topics
print('Inferring topics...')
topic_model.infer_topics(num_topics=15)
# Save model on disk
ut.save_topic_model(topic_model, 'output/NMF_15topics.pickle')

# Print results
print('\nTopics:')
topic_model.print_topics(num_words=10)
print('\nTopic distribution for document 0:',
      topic_model.topic_distribution_for_document(0))
print('\nMost likely topic for document 0:',
      topic_model.most_likely_topic_for_document(0))
print('\nFrequency of topics:',
      topic_model.topics_frequency())
print('\nTop 10 most relevant words for topic 2:',
      topic_model.top_words(2, 10))
