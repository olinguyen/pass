import numpy as np
import os.path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nlp.embeddings import WordEmbedding
from database.utils import get_train_test_data

GLOVE_DIR = "/media/data/glove/glove.6B"
GLOVE_DIR = "/home/olivier/glove/glove.6B"
GLOVE_DIR = "/home/vmadmin/hdbc/glove"
GLOVE_FILENAME = "glove.6B.50d.txt"
GLOVE_FILENAME = "glove.twitter.27B.200d.txt"

class Glove(WordEmbedding):
    def __init__(self):
        pass

    @classmethod
    def load(cls, path=None):
        instance = Glove()
        if path is None:
           path = GLOVE_DIR

        w2v = {}
        embedding_filename = os.path.join(path, GLOVE_FILENAME)
        with open(embedding_filename, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split(" ")

                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                w2v[word] = coefs

        instance.__dict__ = w2v
        print('Found %s word vectors.' % len(w2v))
        return instance

    def get_embedding_matrix(self, tokenizer, max_features=20000, embed_size=200):
        all_embs = np.stack(self.__dict__.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        self.emb_mean, self.emb_std = emb_mean, emb_std

        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = self.__dict__.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_dict(self):
        return self.__dict__


if __name__ == "__main__":
    max_features, embed_size, maxlen = 20000, 50, 75
    X_train, _, _, _= get_train_test_data(merge=True)
    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X_train))
    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(list_tokenized_train, maxlen)

    glove = Glove.load()
    embedding_matrix = glove.get_embedding_matrix(tokenizer, max_features, embed_size)
