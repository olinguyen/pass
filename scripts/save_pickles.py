from keras.preprocessing.text import Tokenizer

import pickle
import numpy as np
import argparse
import os
import pandas as pd

from database.utils import get_train_test_data
from nlp.glove import Glove


MAX_FEATURES = 20000
EMBED_SIZE = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saving and pickling all necessary files')

    parser.add_argument('-v', '--w2v', action='store_true',
                        help='Save w2v dictionary')

    parser.add_argument('-i', '--index', action='store_true',
                        help='Save word index dictionary')

    parser.add_argument('-t', '--tfidf', action='store_true',
                        help='Save TfidfVectorizer object')


    parser.add_argument('-o', '--output_dir', default="models/",
                        help='Path to the output folder')

    args = parser.parse_args()


    if args.w2v:
        glove = Glove.load()
        w2v = glove.get_dict()
        print('w2v dict size:', len(w2v))
        with open(os.path.join(args.output_dir, 'w2v_full.pkl'), 'wb') as f:
            pickle.dump(w2v, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.index:
        X_train, y_train, X_test, y_test = get_train_test_data(merge=True)
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        X_data = pd.concat((X_train, X_test), ignore_index=True)
      
        tokenizer.fit_on_texts(X_data)
        print("Word index dict size:", len(tokenizer.word_index))
        outfile = os.path.join(args.output_dir, 'word_index_full.pkl')
        print("...wrote to", outfile)
        with open(outfile, 'wb') as f:
            pickle.dump(tokenizer.word_index, f, protocol=pickle.HIGHEST_PROTOCOL)
      
        if not args.w2v:
            W2V_DICT_PATH = './models/w2v.pkl'
            with open(W2V_DICT_PATH, 'rb') as f:
                w2v = pickle.load(f)

            embedding_matrix = Glove.get_embedding_matrix_static(w2v,
                                                                 tokenizer,
                                                                 MAX_FEATURES,
                                                                 EMBED_SIZE)
        else:
            embedding_matrix = glove.get_embedding_matrix(tokenizer,
                                                          MAX_FEATURES,
                                                          EMBED_SIZE)

        print("Embedding matrix shape:", embedding_matrix.shape)
        outfile = os.path.join(args.output_dir, 'embedding_matrix_full')
        print("...wrote to", outfile)

        np.save(outfile, embedding_matrix)

