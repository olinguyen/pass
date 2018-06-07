import numpy as np
import pandas as pd
import pickle
import dill
import time
from collections import Counter

from database.query import DataAccess
from bson.objectid import ObjectId
from feature_extraction.transformers import *
from database.utils import get_train_test_data

from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    ensemble_indicators = ['sleep_ensemble_latest',
                           'physical_activity_ensemble_latest',
                           'sedentary_behaviour_ensemble_latest']
    ensemble_models = []

    for indicator in ensemble_indicators:
        with open('./models/%s.pkl' % indicator, 'rb') as f:
            clf = dill.load(f)
            ensemble_models.append((clf, indicator))

    train_test_data = get_train_test_data()

    results = {}

    for _, _, X_test, y_test, indicator in train_test_data:
        results[indicator] = {}
        index = [
            indicator for _,
            indicator in ensemble_models].index(
            indicator +
            "_ensemble_latest")
        clf = ensemble_models[index][0]
        y_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        print(indicator, auc)

        results[indicator]['y_proba'] = y_proba[:, 1]
        results[indicator]['y_pred'] = y_pred
        results[indicator]['y_true'] = y_test
        results[indicator]['text'] = X_test

        result = pd.DataFrame(results[indicator])
        false_pos = (result.y_pred == 1) & (result.y_true == 0)
        false_neg = (result.y_pred == 0) & (result.y_true == 1)

        result.loc[false_neg].to_csv(
            './results/%s_false_neg.csv' %
            indicator, index=False, sep='\t')
        result.loc[false_pos].to_csv(
            './results/%s_false_pos.csv' %
            indicator, index=False, sep='\t')
