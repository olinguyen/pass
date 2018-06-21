from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from xgboost import XGBClassifier

from evaluation.metrics import class_report
from nlp.glove import Glove
from database.utils import get_labeled_data, get_train_test_data
from models.nn_models import *
from pipelines.feature_extractor import get_feature_extractor
from pipelines.models import get_ensemble_model, get_basic_model, get_keras_model


from copy import deepcopy
import numpy as np
import pandas as pd
import pprint
import pickle
import time

EMBED_SIZE = 200 # Word embedding dimensions
MAX_FEATURES = 20000 # Dictionary size
MAX_LEN = 75 # Max number of words in a tweet

W2V_DICT_PATH = 'models/w2v.pkl'

if __name__ == "__main__":

    ps = time.time()

    Xr_train, y_train, Xr_test, y_test = get_train_test_data(merge=True)

    if W2V_DICT_PATH:
        with open(W2V_DICT_PATH, "rb") as f:
            w2v = pickle.load(f)
            print('...loaded w2v dict')
    else:
        glove = Glove.load()
        w2v = glove.get_dict()

    ensemble = get_ensemble_model(w2v)
    #ensemble.steps = ensemble.steps[2:]
    feature_extractor = get_feature_extractor(w2v)
    n_jobs = 8

    cols_target = ['label_pa', 'label_sb', 'label_sleep']

    models = [#("lr", LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', n_jobs=n_jobs)),
#              ("nb", BernoulliNB(alpha=5.0)),
#              ("rf", RandomForestClassifier(n_estimators=300,
#                                             max_depth=10,
#                                             min_samples_split=5,
#                                             n_jobs=n_jobs)),
#              ("xgb", XGBClassifier(n_estimators=150,
#                                     max_depth=8,
#                                     n_jobs=n_jobs)),
#              ("ensemble", ensemble),
              #("svm", SVC(C=100, gamma=0.0001, probability=True)),
              ("nn_lstm", get_keras_model(get_lstm_model)),
              ("nn_cnn", get_keras_model(get_cnn_model)),
             ]

    skf = StratifiedKFold(n_splits=5, random_state=42)

    scoring = {"roc_auc": "roc_auc",
               "accuracy": "accuracy",
               "precision": "precision",
               "recall": "recall",
               "f1score": "f1",
               "average_precision": "average_precision"}

    X_data = pd.concat((Xr_train, Xr_test), ignore_index=True)
    y_data = pd.concat((y_train, y_test), ignore_index=True)

    cv_results = {}
    results_by_class_list = []

    for col in cols_target:
        print(col)
        cv_results[col] = {}
        for model_name, clf in models:
            if model_name != 'ensemble' and 'nn_' not in model_name:
                clf = get_basic_model(clf, w2v)
            cv_result = cross_validate(clf, X_data, y_data.loc[:, col],
                           cv=skf, scoring=scoring, return_train_score=True)
            cv_results[col][model_name] = pd.DataFrame(cv_result).mean()
        results_df = pd.DataFrame(cv_results[col]).T
        results_by_class_list.append(results_df)
        results_df.to_csv('./results/%s_5cv_results.csv' % col, sep='\t')

    pe = time.time()
    print("Completed evaluation in %.2f seconds" % (pe - ps))

    total_results_df = pd.DataFrame(pd.concat(results_by_class_list, keys=[col for col in cols_target]))
    total_results_df.to_csv('./results/total_cv_results.csv', sep='\t')
