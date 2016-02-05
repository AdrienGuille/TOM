# TOM

TOM (TOpic Modeling) is a Python 2.7 library for topic modeling and browsing. Its objective is to allow for an efficient analysis of a text corpus from start to finish, via the discovery of latent topics. To this end, TOM features functions for preparing and vectorizing a text corpus. It also offers a common interface for two topic models (namely LDA using either variational inference or Gibbs sampling, and NMF using alternating least-square with a projected gradient method), and implements three state-of-the-art methods for estimating the optimal number of topics to model a corpus. What is more, TOM constructs an interactive Web-based browser that makes it easy to explore a topic model and the related corpus.

## Installation

We recommend you to install Anaconda (https://www.continuum.io) which will automatically install most of the required dependencies (i.e. pandas, numpy, scipy, scikit-learn, matplotlib, nltk). You should then install the gensim module (https://anaconda.org/anaconda/gensim) and install nltk data (http://www.nltk.org/data.html). 
If you intend to use the French lemmatizer, you should also install MElt on your system (https://www.rocq.inria.fr/alpage-wiki/tiki-index.php?page=MElt).
Eventually, clone or download this repo and run the following command:

```
python setup.py install
```

## Usage

###Load and prepare a text corpus

The following code snippet shows how to load a corpus of French documents, lemmatize them and vectorize them using tf-idf with unigrams.

```
corpus = Corpus(source_file_path='input/raw_corpus.csv',
                language='french', 
                vectorization='tfidf', 
                n_gram=1,
                max_relative_frequency=0.8, 
                min_absolute_frequency=4,
                preprocessor=FrenchLemmatizer())
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
print 'Vector for document 0:\n', corpus.vector_for_document(0)
```

The following code snippet show how to load a corpus without any preprocessing.

```
corpus = Corpus(source_file_path='input/raw_corpus.csv',
                vectorization='tf', 
                preprocessor=None)
```

### Instantiate a topic model and estimate the optimal number of topics

Here, we instantiate a NMF based topic model and generate plots with the three metrics for estimating the optimal number of topics to model the loaded corpus.

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

### Fit a topic model and save/load it

To allow reusing previously learned topics models, TOM can save them on disk, as shown below.

```
topic_model.infer_topics(num_topics=15)
utils.save_topic_model(topic_model, 'output/NMF_15topics.tom')
topic_model = utils.load_topic_model('output/NMF_15topics.tom')
```

### Print information about a topic model

This code excerpt illustrates how one can manipulate a topic model, e.g. get the topic distribution for a document or the word distribution for a topic.

```
print '\nTopics:', topic_model.print_topics(num_words=10)
print '\nTopic distribution for document 0:', \
    topic_model.topic_distribution_for_document(0)
print '\nMost likely topic for document 0:', \
    topic_model.most_likely_topic_for_document(0)
print '\nFrequency of topics:', \
    topic_model.topics_frequency()
print '\nTop 10 most relevant words for topic 2:', \
    topic_model.top_words(2, 10)
```

## Topic model browser: screenshots

### Topic cloud
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/topic_cloud.jpg)
### Topic details
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/topic_0.jpg)
### Document details
![](http://mediamining.univ-lyon2.fr/people/guille/tom-resources/document_31.jpg)
