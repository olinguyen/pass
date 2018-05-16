import os
from feature_extraction.transformers import *
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

nltk_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
GLOVE_DIR = "/media/data/glove/glove.6B"
GLOVE_DIR = "/home/vmadmin/hdbc/glove"

def get_glove_w2v(glove_path=GLOVE_DIR):
    w2v = {}
    with open(os.path.join(glove_path, 'glove.6B.300d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            w2v[word] = coefs

    print('Found %s word vectors.' % len(w2v))
    return w2v

def get_features(w2v=None):
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

                # TF-IDF over tokens
                ('tfidf_token_ngrams', TfidfVectorizer(ngram_range=(1,4),
                           lowercase=False,
                           tokenizer=nltk_tokenizer.tokenize,
                           stop_words='english',
                           min_df=3,
                           max_df=0.9,
                           strip_accents='unicode',
                           use_idf=True,
                           sublinear_tf=True
                          )),

                # TF-IDF over characters
                ('tfidf_token_chars', TfidfVectorizer(ngram_range=(1,4),
                           lowercase=False,
                           analyzer='char',
                           min_df=3,
                           max_df=0.9,
                           use_idf=True,
                           sublinear_tf=True
                          )),
            ],)

if __name__ == "__main__":
    wl = Pipeline([('feats', get_features())])
    y = wl.fit_transform(["hello world"], [0])
    print(y.shape) 
