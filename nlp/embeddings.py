import numpy as np
import os.path

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

def get_embedding_matrix(max_features, tokenizer, w2v, embed_size):
    all_embs = np.stack(w2v.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    emb_mean, emb_std

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = w2v.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix
