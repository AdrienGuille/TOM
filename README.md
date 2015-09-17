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

- Vectorize documents and infer topics with LDA, LSA or NMF:
```
lda = LDA(raw_data=documents, language='french')
print 'vocabulary size:', lda.get_vocabulary_size()
lda.infer_topics(num_topics=15)
print '\nLDA:'
lda.print_topics(num_words=10)
```
```
lsa = LSA(vectorized_data=lda)
lsa.infer_topics(num_topics=15)
print '\nLSA:'
lsa.print_topics(num_words=10)
```
```
nmf = NMF(vectorized_data=lda)
nmf.infer_topics(num_topics=15)
print '\nNMF:'
nmf.print_topics(num_words=10)
```
