# coding: utf-8
from nlp.lemmatizer import FrenchLefff
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from structure.corpus import Corpus
from flask import Flask, render_template
import utils
import shutil
import os

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

# Flask Web server
app = Flask(__name__)

# Load corpus
corpus = Corpus(source_file_path='../input/egc.csv',
                language='french',
                max_relative_frequency=0.8,
                min_absolute_frequency=4,
                lemmatizer=None)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)

# Infer topics
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=15)

# Clean the data directory
if os.path.exists('static/data'):
    shutil.rmtree('static/data')
os.makedirs('static/data')

# Export topic cloud
utils.save_topic_cloud(topic_model, 'static/data/topic_cloud.json')

# Export details about topics
for topic_id in range(topic_model.nb_topics):
    utils.save_word_distribution(topic_model.top_words(topic_id, 20), 'static/data/word_distribution'+str(topic_id)+'.tsv')
    evolution = []
    for i in range(2004, 2016):
        evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, 'static/data/frequency'+str(topic_id)+'.tsv')

# Export details about documents
for doc_id in range(topic_model.corpus.size):
    utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id), 'static/data/topic_distribution_d'+str(doc_id)+'.tsv')

# Export details about words
for word_id in range(len(topic_model.corpus.vocabulary)):
    utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id), 'static/data/topic_distribution_w'+str(word_id)+'.tsv')

# Affiliate documents with topics
topic_affiliations = topic_model.documents_per_topic()

# Export per-topic author network
# for topic_id in range(topic_model.nb_topics):
#     utils.save_author_network(corpus.collaboration_network(topic_affiliations[topic_id]), 'static/data/author_network'+str(topic_id)+'.json')


@app.route('/')
def index():
    return render_template('index.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/topic_cloud.html')
def topic_cloud():
    return render_template('topic_cloud.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(corpus.vocabulary)):
        word_list.append((i, corpus.get_word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(corpus.vocabulary)/5)
    print words_per_column, 'words per column'
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        print sub_vocabulary
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@app.route('/topic/<tid>.html')
def topic_details(tid):
    ids = topic_affiliations[int(tid)]
    documents = []
    for document_id in ids:
        documents.append((corpus.short_content(document_id).capitalize(),
                          corpus.authors(document_id),
                          corpus.date(document_id), document_id))
    documents.pop(0)
    return render_template('topic.html',
                           topic_id=tid,
                           frequency=round(topic_model.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size))


@app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((corpus.word_for_id(a_word_id), vector[a_word_id], a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    return render_template('document.html',
                           doc_id=did,
                           words=word_list[:25],
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           authors=corpus.authors(int(did)),
                           year=corpus.date(int(did)))


@app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in corpus.docs_for_word(int(wid)):
        documents.append((corpus.short_content(document_id).capitalize(),
                          corpus.authors(document_id),
                          corpus.date(document_id), document_id))
    return render_template('word.html',
                           word_id=wid,
                           word=topic_model.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(corpus.size),
                           documents=documents)

if __name__ == '__main__':
    # Load corpus
    app.run(debug=True, host='localhost', port=2016)
