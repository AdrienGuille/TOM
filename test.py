# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from nlp.lemmatizer import FrenchLefff, EnglishWordNet
from structure.corpus import Corpus
from visualization.visualization import Visualization
import stats
import pickle
import utils

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

print 'Load documents from CSV'
corpus = Corpus(source_file_path='input/egc.csv',
                language='french',
                max_relative_frequency=0.8,
                min_absolute_frequency=4,
                lemmatizer=FrenchLefff())
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=20)
print '\nTopics:'
topic_model.print_topics(num_words=10)
print '\nDocument 2:', topic_model.corpus.full_content(2)
print '\nVector representation of document 2:\n', topic_model.corpus.vector_for_document(2)
print '\nTopic distribution for document 2:', topic_model.topic_distribution_for_document(2)
print '\nMost likely topic for document 2:', topic_model.most_likely_topic_for_document(2)
print '\nTopics frequency:', topic_model.topics_frequency()
print '\nTopic 2 frequency:', topic_model.topic_frequency(2)
print '\nTop 10 most likely words for topic 2:', topic_model.top_words(2, 10)