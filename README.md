# TopicModeling

A python program for topic modeling.

## Topic models

- Latent Dirichlet Allocation (LDA)
- Latent Semantic Analysis (LSA)
- Non-negative Matrix Factorization (NMF)

## Dependencies

- NLTK
- Scikit-Learn
- Gensim

## Documentation

- Read data from a text file:
```
with open('data/test.txt') as f:
    documents = f.read().splitlines()
print 'corpus size:', len(documents)
```
- Vectorize documents (a list of documents (document=string)) and infer topics with LDA:
```
lda = LatentDirichletAllocation(raw_data=documents, language='french')
print 'vocabulary size:', lda.get_vocabulary_size()
lda.infer_topics(num_topics=15)
```
- Print per-topic word distributions:
```
print '\nLDA:'
lda.print_topics(num_words=10)
```
- Print topic distribution for a specific document (the id corresponds to the index in the input list):
```
print '\nTopic distribution for document id=1:'
lda.print_topics_for_document(1)
```
- Same thing with LSA:
```
lsa = LatentSemanticAnalysis(vectorized_data=lda)
lsa.infer_topics(num_topics=15)
print '\nLSA:'
lsa.print_topics(num_words=10)
print '\nTopic distribution for document id=1:'
lsa.print_topics_for_document(1)
```
- Same thing with NMF:
```
nmf = NonNegativeMatrixFactorization(vectorized_data=lda)
nmf.infer_topics(num_topics=15)
print '\nNMF:'
nmf.print_topics(num_words=10)
print '\nTopic distribution for document id=1:'
nmf.print_topics_for_document(1)
```
