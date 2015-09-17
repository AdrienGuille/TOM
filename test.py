# coding: utf-8
from nlp.topic_model import LDA, LSA, NMF

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

with open('data/test.txt') as f:
    documents = f.read().splitlines()
print 'corpus size:', len(documents)

lda = LDA(raw_data=documents, language='french')
print 'vocabulary size:', lda.get_vocabulary_size()
lda.infer_topics(num_topics=15)
print '\nLDA:'
lda.print_topics(num_words=10)

lsa = LSA(vectorized_data=lda)
lsa.infer_topics(num_topics=15)
print '\nLSA:'
lsa.print_topics(num_words=10)

nmf = NMF(vectorized_data=lda)
nmf.infer_topics(documents)
print '\nNMF:'
nmf.print_topics(num_words=10)
