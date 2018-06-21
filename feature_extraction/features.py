import os
from feature_extraction.transformers import *
from feature_extraction.nbsvm import NBFeaturer
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp.glove import Glove

nltk_tokenizer = TweetTokenizer(
    strip_handles=True,
    reduce_len=True,
    preserve_case=False)


def get_features(w2v=None):
    tfidf_words = TfidfVectorizer(ngram_range=(1, 4),
                                  max_features=5000,
                                  lowercase=True,
                                  tokenizer=nltk_tokenizer.tokenize,
                                  stop_words='english',
                                  min_df=3,
                                  max_df=0.9,
                                  strip_accents='unicode',
                                  use_idf=True,
                                  norm='l2',
                                  sublinear_tf=True)

    tfidf_chars = TfidfVectorizer(ngram_range=(1, 4),
                                  max_features=5000,
                                  lowercase=False,
                                  analyzer='char',
                                  min_df=3,
                                  max_df=0.9,
                                  use_idf=True,
                                  norm='l2',
                                  sublinear_tf=True)

    if not w2v:
        glove = Glove.load()
        w2v = glove.get_dict()

    return FeatureUnion([
        # Average length of word in a sentence
        ('avg_word_len', AverageWordLengthExtractor()),

        # Number of words
        ('num_words', NumWordExtractor()),

        # Number of characters in a sentence
        ('num_chars', CharLengthExtractor()),

        # Number of unique words used
        ('num_unique', NumUniqueWordExtractor()),

        # Naive bayes tfidf features
        ("tfidf_nbf", Pipeline([
            ("wc_tfidf", FeatureUnion([
                # TF-IDF over tokens
                ('tfidf_token_ngrams', tfidf_words),
                # TF-IDF over characters
                ('tfidf_token_chars', tfidf_chars)
                ])),

            ("nbf", NBFeaturer(alpha=10))
        ])),

        # Averaged word embedding, weighted by tfidf
        ('w2v', TfidfEmbeddingVectorizer(w2v))

        # Averaged word embedding
        #('w2v', MeanEmbeddingVectorizer(w2v))
    ])

if __name__ == "__main__":
    wl = Pipeline([('feats', get_features())])
    y = wl.fit_transform(["hello world"], [0])
    print(y.shape)
