# TOM

TOM (TOpic Modeling) is a python library for topic modeling and exploration. It features functions to preprocess (stemming, lemmatizing, vectorizing) text as well as functions to estimate the optimal number of topics. It provides a common interface for several topic models (LSA, LDA, NMF). It also features a Web-based topic/document explorer (see screenshots below).

## Documentation

- Import required classes and static functions:
```
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from nlp.preprocessor import FrenchLemmatizer, EnglishStemmer
from structure.corpus import Corpus
from visualization.visualization import Visualization
```
- Load a Corpus instance with a collection of documents stored in a csv file:
```
# Load and prepare a corpus
print 'Load documents from CSV'
corpus = Corpus(source_file_path='input/egc.csv',
                language='french',  # determines the stop words
                vectorization='tf',  # 'tf' (term-frequency) or 'tfidf' (term-frequency inverse-document-frequency)
                max_relative_frequency=0.8,  # ignore words which relative frequency is > than max_relative_frequency
                min_absolute_frequency=4,  # ignore words which absolute frequency is < than min_absolute_frequency
                preprocessor=None)  # determines how documents are preprocessed (e.g. stemming, lemmatization)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
print 'Vector representation of document 2:\n', corpus.vector_for_document(2)
```
- Instantiate a topic model
```
topic_model = LatentDirichletAllocation(corpus=corpus)
```
- Estimate the optimal number of topics
```
viz = Visualization(topic_model)
viz.plot_arun_metric(10, 30, 5, '/Users/adrien/Desktop/arun.png')
```
- Infer topics
```
topic_model.infer_topics(num_topics=20)
```
- Print some results
```
print '\nTopics:'
topic_model.print_topics(num_words=10)
print '\nDocument 2:', topic_model.corpus.full_content(2)
print '\nTopic distribution for document 2:', topic_model.topic_distribution_for_document(2)
print '\nMost likely topic for document 2:', topic_model.most_likely_topic_for_document(2)
print '\nTopics frequency:', topic_model.topics_frequency()
print '\nTopic 2 frequency:', topic_model.topic_frequency(2)
print '\nTop 10 most likely words for topic 2:', topic_model.top_words(2, 10)
```

## Topic/document explorer

### Topic cloud
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/topic_cloud.jpg)
### Topic details
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/topic_details.jpg)
### Document details
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/document_details.jpg)
