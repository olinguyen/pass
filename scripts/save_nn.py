import sys, os, re, csv, codecs
import numpy as np, pandas as pd
import time, pickle, joblib


from nlp.glove import Glove
from database.utils import get_train_test_data
from models.nn_models import *
from pipelines.models import get_keras_model

W2V_DICT_PATH = 'models/w2v.pkl'


if __name__ == "__main__":
    if W2V_DICT_PATH:
        with open(W2V_DICT_PATH, "rb") as f:
            w2v = pickle.load(f)
            print('...loaded w2v dict')
    else:
        glove = Glove.load()
        w2v = glove.get_dict()

    model = get_keras_model(get_lstm_model)

    Xr_train, y_train, Xr_test, y_test = get_train_test_data(merge=True)

    X_data = pd.concat((Xr_train, Xr_test), ignore_index=True)
    y_data = pd.concat((y_train, y_test), ignore_index=True)

    model.fit(X_data, y_data)
    joblib.dump(model, './models/lstm.pkl')
