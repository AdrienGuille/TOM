# coding: utf-8
from gensim import matutils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


class Corpus:

    def __init__(self,
                 text_file_path,
                 time_file_path=None,
                 language=None,
                 max_relative_frequency=0.95,
                 min_absolute_frequency=2):
        with open(text_file_path) as f:
            self.documents = f.read().splitlines()
        self.size = len(self.documents)
        self.dates = None
        self.time_index = None
        if time_file_path is not None:
            with open(time_file_path) as f:
                self.dates = f.read().splitlines()
            if self.size != len(self.dates):
                raise ValueError('Text file and time file lengths mismatch')
            self.time_index = {}
            for i in range(0, self.size):
                date = self.dates[i]
                docs = self.time_index.get(date)
                if docs is None:
                    docs = [i]
                else:
                    docs.append(i)
                self.time_index[date] = docs
        stop_words = []
        if language is not None:
            stop_words = stopwords.words(language)
        self.vectorizer = TfidfVectorizer(max_df=max_relative_frequency,
                                          min_df=min_absolute_frequency,
                                          max_features=2000,
                                          stop_words=stop_words)
        self.sklearn_tfidf = self.vectorizer.fit_transform(self.documents)
        self.gensim_tfidf = matutils.Sparse2Corpus(self.sklearn_tfidf, documents_columns=False)
        vocab = self.vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def get_ids(self, date):
        if self.time_index is not None:
            return self.time_index.get(str(date))
        else:
            raise ValueError('No temporal information available for this corpus')

    def get_vector_for_document(self, doc_id):
        return self.sklearn_tfidf[doc_id]

    def get_word_for_id(self, word_id):
        return self.vocabulary.get(word_id)
