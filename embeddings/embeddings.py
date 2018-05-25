import numpy as np

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
