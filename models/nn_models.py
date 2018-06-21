from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D, Convolution1D, MaxPooling1D
from keras.models import Model, Sequential

import numpy as np
import os

from nlp.glove import Glove

EMBED_MATRIX_PATH = os.path.join(
            os.path.dirname(__file__),
            '../models/embedding_matrix_full.npy')

def get_lstm_model(embedding_matrix=None, maxlen=75):

    if embedding_matrix is None:
        print("Loading embedding_matrix")
        embedding_matrix = np.load(EMBED_MATRIX_PATH)

    embed_size = embedding_matrix.shape[1]

    inp = Input(shape=(maxlen,))
    x = Embedding(
        input_dim=len(embedding_matrix),
        output_dim=embed_size,
        weights=[embedding_matrix],
        trainable=False)(inp)
    x = Bidirectional(
        LSTM(
            50,
            return_sequences=True,
            dropout=0.1,
            recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def get_cnn_model(embedding_matrix=None, maxlen=75):
    if embedding_matrix is None:
        print("Loading embedding_matrix")
        embedding_matrix = np.load(EMBED_MATRIX_PATH)

    embed_size = embedding_matrix.shape[1]

    graph_in = Input(shape=(maxlen, embed_size))
    convs = []
    for fsz in range(1, 4):
        conv = Convolution1D(
            filters=300,
            kernel_size=fsz,
            padding='valid',
            activation='relu')(graph_in)
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    out = concatenate(convs)
    graph = Model(inputs=graph_in, outputs=out)

    model = Sequential()
    model.add(Embedding(input_dim=len(embedding_matrix),
                        output_dim=embed_size,
                        weights=[embedding_matrix],
                        input_length=maxlen))

    model.add(graph)
    model.add(Dense(300))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model
