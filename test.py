# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

with open('data/test.txt') as f:
    documents = f.read().splitlines()
print 'corpus size:', len(documents)

lda = LatentDirichletAllocation(raw_data=documents, language='french')
print 'vocabulary size:', lda.get_vocabulary_size()
lda.infer_topics(num_topics=15)
print '\nLDA:'
lda.print_topics(num_words=10)
print '\nTopic distribution for document id=1:'
lda.print_topics_for_document(1)

lsa = LatentSemanticAnalysis(vectorized_data=lda)
lsa.infer_topics(num_topics=15)
print '\nLSA:'
lsa.print_topics(num_words=10)
print '\nTopic distribution for document id=1:'
lsa.print_topics_for_document(1)

nmf = NonNegativeMatrixFactorization(vectorized_data=lda)
nmf.infer_topics(num_topics=15)
print '\nNMF:'
nmf.print_topics(num_words=10)
print '\nTopic distribution for document id=1:'
nmf.print_topics_for_document(1)
