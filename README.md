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

- Import required classes and static functions:
```
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from structure.corpus import Corpus
import visualization
```
- Load corpus from a text file (one document per line):
```
corpus = Corpus(text_file_path='input/test.txt',
                language='french',
                max_relative_frequency=0.9,
                min_absolute_frequency=2)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
```
- Print raw document (the id corresponds to the line number minus 1) and its vector representation:
```
print '\nDocument 2:', topic_model.corpus.documents[2]
print '\nVector representation of document 2:\n', topic_model.corpus.get_vector_for_document(2)
```
- Infer topics:
```
topic_model = LatentDirichletAllocation(corpus) #  or LatentSemanticAnalysis(corpus) or NonNegativeMatrixFactorization(corpus)
topic_model.infer_topics(num_topics=5)
```
- Save the model on disk using Pickle:
```
# pickle.dump(topic_model, open('output/lda.pickle', 'wb'))
# topic_model = pickle.load(open('output/lda.pickle', 'rb'))
```
- Print per-topic word distributions:
```
print '\nTopics:'
topic_model.print_topics(num_words=10, display_weights=True)
```
- Print and plot topic distribution for a document:
```
print '\nTopic distribution for document 2:', topic_model.topic_distribution(2)
visualization.plot_topic_distribution(topic_model.topic_distribution(2))
```
- Get the most likely topic for a document:
```
print '\nMost likely topic for document 2:', topic_model.most_likely_topic_for_document(2)
```
