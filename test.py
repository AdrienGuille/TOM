# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from structure.corpus import Corpus
import visualization
import pickle

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


corpus = Corpus(text_file_path='input/test.txt',
                language='french',
                max_relative_frequency=0.9,
                min_absolute_frequency=2)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
topic_model = LatentDirichletAllocation(corpus=corpus)
topic_model.infer_topics(num_topics=5)
# pickle.dump(topic_model, open('output/nmf.pickle', 'wb'))
# topic_model = pickle.load(open('output/lda.pickle', 'rb'))

print '\nTopics:'
topic_model.print_topics(num_words=10, display_weights=True)
print '\nDocument 2:', topic_model.corpus.documents[2]
print '\nVector representation of document 2:\n', topic_model.corpus.get_vector_for_document(2)
print '\nTopic distribution for document 2:', topic_model.topic_distribution(2)
visualization.plot_topic_distribution(topic_model.topic_distribution(2))
print '\nMost likely topic for document 2:', topic_model.most_likely_topic_for_document(2)
print '\nTopic absolute frequency:', topic_model.topic_absolute_frequency()
print '\nTopic relative frequency:', topic_model.topic_relative_frequency()
