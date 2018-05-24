import os
from feature_extraction.transformers import *
from feature_extraction.nbsvm import NBFeaturer
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

nltk_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
GLOVE_DIR = "/media/data/glove/glove.6B"
GLOVE_DIR = "/home/vmadmin/hdbc/glove"
GLOVE_FILENAME = "glove.6B.300d.txt"
GLOVE_FILENAME = "glove.twitter.27B.200d.txt"

def get_glove_w2v(glove_path=GLOVE_DIR):
    w2v = {}
    embedding_filename = os.path.join(glove_path, GLOVE_FILENAME)
    with open(embedding_filename, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split(" ")

            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
#coefs = np.array([float(i) for i in values[1:] if len(i) > 1])
            w2v[word] = coefs

    print('Found %s word vectors.' % len(w2v))
    return w2v

def get_features(w2v=None):
    tfidf_words = TfidfVectorizer(ngram_range=(1,4),
                           lowercase=False,
                           tokenizer=nltk_tokenizer.tokenize,
                           stop_words='english',
                           min_df=3,
                           max_df=0.9,
                           strip_accents='unicode',
                           use_idf=True,
                           sublinear_tf=True)

    tfidf_chars = TfidfVectorizer(ngram_range=(1,4),
                           lowercase=False,
                           analyzer='char',
                           min_df=3,
                           max_df=0.9,
                           use_idf=True,
                           sublinear_tf=True)

    if not w2v:
        w2v = get_glove_w2v()
    return FeatureUnion([
                # Average length of word in a sentence
                ('avg_word_len', AverageWordLengthExtractor()),

                # Number of words
                ('num_words', NumWordExtractor()),

                # Number of characters in a sentence
                ('num_chars', CharLengthExtractor()),

                # Number of unique words used
                ('num_unique', NumUniqueWordExtractor()),

                # Averaged word embedding, weighted by tfidf
                ('w2v', TfidfEmbeddingVectorizer(w2v)),

                ("tfidf_nbf", Pipeline([
                    ("wc_tfidf", FeatureUnion([
                          # TF-IDF over tokens
                          ('tfidf_token_ngrams', tfidf_words),
                          # TF-IDF over characters
                          ('tfidf_token_chars', tfidf_chars)])),

                    ("nbf", NBFeaturer(alpha=10))
                ]))

            ])

if __name__ == "__main__":
    wl = Pipeline([('feats', get_features())])
    y = wl.fit_transform(["hello world"], [0])
    print(y.shape) 
