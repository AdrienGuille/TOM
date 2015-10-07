# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from nlp.preprocessor import FrenchLemmatizer, EnglishStemmer, EnglishLemmatizer
from structure.corpus import Corpus
from visualization.visualization import Visualization
import utils

__author__ = "Adrien Guille, Pavel Soriano"
__email__ = "adrien.guille@univ-lyon2.fr"

# Load and prepare a corpus
print 'Load documents from CSV'
corpus = Corpus(source_file_path='input/egc_lemmatized.csv',
                language='french',  # language for stop words
                vectorization='tfidf',  # 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
                max_relative_frequency=0.8,  # ignore words which relative frequency is > than max_relative_frequency
                min_absolute_frequency=4,  # ignore words which absolute frequency is < than min_absolute_frequency
                preprocessor=None)  # pre-process documents
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
print 'Vector representation of document 0:\n', corpus.vector_for_document(0)

# Instantiate a topic model
topic_model = NonNegativeMatrixFactorization(corpus=corpus)

# Estimate the optimal number of topics
viz = Visualization(topic_model)
viz.plot_greene_metric(min_num_topics=10, max_num_topics=30, tao=10, step=1, top_n_words=10)
viz.plot_arun_metric(min_num_topics=5, max_num_topics=30, iterations=5)

# Infer topics
print 'Inferring topics...'
topic_model.infer_topics(num_topics=15)
# Save model on disk
utils.save_topic_model(topic_model, 'output/LDA_egc.pickle')

# Print results
print '\nTopics:'
topic_model.print_topics(num_words=10)
print '\nTopic distribution for document 0:', topic_model.topic_distribution_for_document(0)
print '\nMost likely topic for document 0:', topic_model.most_likely_topic_for_document(0)
print '\nTopics frequency:', topic_model.topics_frequency()
print '\nTopic 2 frequency:', topic_model.topic_frequency(2)
print '\nTop 10 most likely words for topic 2:', topic_model.top_words(2, 10)
