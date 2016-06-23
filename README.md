# TOM

TOM (TOpic Modeling) is a Python 3 library for topic modeling and browsing, licensed under the MIT license. Its objective is to allow for an efficient analysis of a text corpus from start to finish, via the discovery of latent topics. To this end, TOM features functions for preparing and vectorizing a text corpus. It also offers a common interface for two topic models (namely LDA using either variational inference or Gibbs sampling, and NMF using alternating least-square with a projected gradient method), and implements three state-of-the-art methods for estimating the optimal number of topics to model a corpus. What is more, TOM constructs an interactive Web-based browser that makes it easy to explore a topic model and the related corpus.

## Installation

We recommend you to install Anaconda (https://www.continuum.io) which will automatically install most of the required dependencies (i.e. pandas, numpy, scipy, scikit-learn, matplotlib, flask). You should then install the lda module (pip install lda).
Eventually, clone or download this repo and run the following command:

```
python setup.py install
```

Or, install it directly from PyPi:

```
pip install tom_lib
```

## Usage

We provide two sample programs, topic_model.py (which shows you how to load and prepare a corpus, estimate the optimal number of topics, infer the topic model and then manipulate it) and topic_model_browser.py (which shows you how to generate a topic model browser to explore a corpus), to help you get started using TOM.

###Load and prepare a textual corpus

The following code snippet shows how to load a corpus of French documents and vectorize them using tf-idf with unigrams.

```
corpus = Corpus(source_file_path='input/raw_corpus.csv',
                language='french', 
                vectorization='tfidf', 
                n_gram=1,
                max_relative_frequency=0.8, 
                min_absolute_frequency=4)
print('corpus size:', corpus.size)
print('vocabulary size:', len(corpus.vocabulary))
print('Vector representation of document 0:\n', corpus.vector_for_document(0))
```

### Instantiate a topic model and infer topics

It is possible to instantiate a NMF or LDA object then infer topics. 

NMF:

```
topic_model = NonNegativeMatrixFactorization(corpus)
topic_model.infer_topics(num_topics=15)
```

LDA (using either the standard variational Bayesian inference or Gibbs sampling):

```
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='variational')
```
```
topic_model = LatentDirichletAllocation(corpus)
topic_model.infer_topics(num_topics=15, algorithm='gibbs')
```

### Instantiate a topic model and estimate the optimal number of topics

Here we instantiate a NMF object, then generate plots with the three metrics for estimating the optimal number of topics.

```
topic_model = NonNegativeMatrixFactorization(corpus)
viz = Visualization(topic_model)
viz.plot_greene_metric(min_num_topics=5, 
                       max_num_topics=50, 
                       tao=10, step=1, 
                       top_n_words=10)
viz.plot_arun_metric(min_num_topics=5, 
                     max_num_topics=50, 
                     iterations=10)
viz.plot_brunet_metric(min_num_topics=5, 
                       max_num_topics=50,
                       iterations=10)
```

### Save/load a topic model

To allow reusing previously learned topics models, TOM can save them on disk, as shown below.

```
utils.save_topic_model(topic_model, 'output/NMF_15topics.tom')
topic_model = utils.load_topic_model('output/NMF_15topics.tom')
```

### Print information about a topic model

This code excerpt illustrates how one can manipulate a topic model, e.g. get the topic distribution for a document or the word distribution for a topic.

```
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
```

## Topic model browser: screenshots

### Topic cloud
![](http://mediamining.univ-lyon2.fr/people/guille/tom_resources/topic_cloud.jpg)
### Topic details
![](http://mediamining.univ-lyon2.fr/people/guille/tom_resources/topic_0.jpg)
### Document details
![](http://mediamining.univ-lyon2.fr/people/guille/tom_resources/document_31.jpg)
