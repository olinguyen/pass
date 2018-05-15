import numpy as np
from tokenizer import tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

class TextCleanExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.T = tokenizer.TweetTokenizer(preserve_handles=False,
                                          preserve_url=False,
                                          preserve_len=False,
                                          preserve_hashes=False,
                                          preserve_case=False)

    def transform(self, X):
        return [' '.join(self.T.tokenize(sentence)) for sentence in X]

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return dict(T = self.T)

class NumWordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(sentence.split()) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return dict()

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def average_word_length(self, text):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in text])

    def transform(self, X):
        return np.array([self.average_word_length(sentence.split()) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return dict()

class CharLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(sentence) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return dict()


class NumUniqueWordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X):
        return np.array([len(Counter(sentence.split())) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
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
        return dict()

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(self.word2vec.values())))

    def fit(self, X, y):
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
    def __init__(self, word2vec):

        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(next(iter(self.word2vec.values())))
        else:
            self.dim = 0
        self.tfidf = None


    def fit(self, X, y):
        tfidf = TfidfVectorizer(#ngram_range=(1,4),
                                lowercase=True,
                                analyzer='word',
                                #tokenizer=tokenize,
                                #stop_words='english'
                                )
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in sentence.split() if w in self.word2vec] or
                         [np.zeros(self.dim)], axis=0)
                         for sentence in X])


    def get_params(self, deep=True):
        return dict(word2vec = self.word2vec)

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

