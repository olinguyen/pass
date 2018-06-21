from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tokenizer import tokenizer

from collections import Counter, defaultdict
import numpy as np
import os
import pickle


class TextCleanExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.T = tokenizer.TweetTokenizer(preserve_handles=False,
                                          preserve_url=False,
                                          preserve_len=False,
                                          preserve_hashes=False,
                                          preserve_emoji=False,
                                          preserve_case=True,
                                          regularize=True)

    def transform(self, X):
        return [' '.join(self.T.tokenize(sentence)) for sentence in X]

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict(T=self.T)


class NumWordExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(sentence.split())
                         for sentence in X]).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict()

class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list 
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, texts, y=None):
        #self.fit_on_texts(texts)
        word_index_path = os.path.join(
            os.path.dirname(__file__),
            '../models/word_index_full.pkl')
        with open(word_index_path, 'rb') as f:
            self.word_index = pickle.load(f)
        return self
    
    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))

class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length. 
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros
    
    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during 
        transform it is transformed to a 0
    """
    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None
        
    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self
    
    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def average_word_length(self, text):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in text])

    def transform(self, X):
        return np.array([self.average_word_length(sentence.split())
                         for sentence in X]).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict()


class CharLengthExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(sentence) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict()


class NumUniqueWordExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(Counter(sentence.split()))
                         for sentence in X]).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return dict()


class ProbExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, models):
        self.models = models

    def transform(self, X, y=None):
        y_probas = []
        for model in self.models:
            y_prob = model.predict_proba(X)[:, 1]
            y_probas.append(y_prob)

        return np.array(y_probas).transpose()

    def fit(self, X, y=None):
        for model in self.models:
            model.fit(X, y)
        return self

    def get_params(self, deep=True):
        return dict(models=self.models)

    def set_params(self, **kwargs):
        return self


class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(self.word2vec.values())))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in sentence.split() if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for sentence in X
        ])

    def get_params(self, deep=True):
        return dict()

# and a tf-idf version of the same


class TfidfEmbeddingVectorizer(object):

    def __init__(self, word2vec, tfidf=None):

        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(next(iter(self.word2vec.values())))
        else:
            self.dim = 0

    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(analyzer=lambda x: x)
        self.tfidf.fit(X)

        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        #print(self.tfidf.idf_.shape)
        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, [
                (w, self.tfidf.idf_[i]) for w, i in self.tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in sentence.split() if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for sentence in X])

    def get_params(self, deep=True):
        return dict(word2vec=self.word2vec)

if __name__ == '__main__':
    wl = NumUniqueWordExtractor()
    test = ["hello world", "this is a test"]
    y = [0, 0]
    wl.fit(test, y)
    wl.fit_transform(test, y)
    out = wl.transform(test)
    print(out)

    cleaner = TextCleanExtractor()
    out = cleaner.transform(["hello @world, this is #olivier!!!!!!$$#$# #$#"])
    print(out)
