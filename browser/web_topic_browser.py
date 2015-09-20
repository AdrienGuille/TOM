# coding: utf-8
from nlp.topic_model import LatentDirichletAllocation, LatentSemanticAnalysis, NonNegativeMatrixFactorization
from structure.corpus import Corpus
from flask import Flask, render_template
import utils
from unidecode  import unidecode

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

# Flask Web server
app = Flask(__name__)

corpus = Corpus(text_file_path='../input/egc/abstracts.txt',
                time_file_path='../input/egc/dates.txt',
                language='french',
                max_relative_frequency=0.8,
                min_absolute_frequency=4)
print 'corpus size:', corpus.size
print 'vocabulary size:', len(corpus.vocabulary)
# Infer topic model
topic_model = NonNegativeMatrixFactorization(corpus=corpus)
topic_model.infer_topics(num_topics=15)
# Export topic cloud
utils.save_topic_cloud(topic_model, 'static/topic_cloud.json')
# Export details about topics
for topic_id in range(topic_model.nb_topics):
    utils.save_word_distribution(topic_model.get_top_words(topic_id, 20), 'static/word_distribution'+str(topic_id)+'.tsv')
    evolution = []
    for i in range(2004, 2016):
        evolution.append((i, topic_model.topic_frequency(topic_id, date=i)))
    utils.save_topic_evolution(evolution, 'static/frequency'+str(topic_id)+'.tsv')
topic_affiliations = topic_model.documents_per_topic()
with open('../input/egc/titles0.txt') as f:
    original_titles = f.read().splitlines()
with open('../input/egc/authors.txt') as f:
    authors = f.read().splitlines()
with open('../input/egc/dates.txt') as f:
    dates = f.read().splitlines()


@app.route('/')
def topic_cloud():
    return render_template('topic_cloud.html')


@app.route('/topic/<tid>')
def topic_details(tid):
    requested_topic = int(str(tid))
    ids = topic_affiliations[requested_topic]
    documents = []
    for doc_id in ids:
        documents.append((unidecode(original_titles[doc_id]), unidecode(authors[doc_id]), dates[doc_id]))
    return render_template('topic.html', topic_id=tid, documents=documents)


if __name__ == '__main__':
    # Load corpus
    app.run(debug=True, host='localhost', port=2016)
