from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from database.utils import get_labeled_data
from pipelines.feature_extractor import get_feature_extractor
from pipelines.models import get_ensemble_model

from copy import deepcopy
import numpy as np
import pandas as pd
import pprint
import time


if __name__ == "__main__":

    ps = time.time()

    ensemble = get_ensemble_model()
    ensemble.steps = ensemble.steps[2:]
    feature_extractor = get_feature_extractor()

    models = [("lr", LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', n_jobs=-1)),
               ("nb", BernoulliNB(alpha=5.0)),
               ("rf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, n_jobs=-1)),
               ("xgb", XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1)),
               ("et", ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=10, n_jobs=-1)),
               ("svm", SVC(C=100, gamma=0.0001, probability=True)),
               ("ensemble", ensemble)]

    results = {}

    for X_raw, y, indicator in data:
        X = feature_extractor.fit_transform(X_raw, y)
        results[indicator] = {}
        for name, classifier in models:
            results[indicator][name] = {}

            cv = StratifiedKFold(n_splits=5, random_state=42)
            scores = []
            conf_mat = np.zeros((2, 2))      # Binary classification
            false_pos = set()
            false_neg = set()
            train_times = []

            for train_i, val_i in cv.split(X, y):
                clf = deepcopy(classifier)
                X_train, X_val = X[train_i], X[val_i]
                y_train, y_val = y[train_i], y[val_i]
                ts = time.time()

                clf.fit(X_train, y_train)
                te = time.time()

                train_times.append(te - ts)

                y_pprobs = clf.predict_proba(X_val)       # Predicted probabilities
                y_plabs = np.squeeze(clf.predict(X_val))  # Predicted class labels

                scores.append(roc_auc_score(y_val, y_pprobs[:, 1]))
                confusion = confusion_matrix(y_val, y_plabs)
                conf_mat += confusion

                # Collect indices of false positive and negatives
                fp_i = np.where((y_plabs==1) & (y_val==0))[0]
                fn_i = np.where((y_plabs==0) & (y_val==1))[0]
                false_pos.update(val_i[fp_i])
                false_neg.update(val_i[fn_i])

            print("\n[%s][%s] Mean score: %0.2f (+/- %0.2f)" % (indicator, name, np.mean(scores), np.std(scores) * 2))
            conf_mat /= 5
            #print("Mean CM: \n", conf_mat)
            #print("\nMean classification measures: \n")
            measures = class_report(conf_mat)
            for metric in measures:
                results[indicator][name][metric] = measures[metric]

            results[indicator][name]['mean_roc_auc'] = np.mean(scores)
            results[indicator][name]['std_roc_auc'] = np.std(scores) * 2
            results[indicator][name]['train_time'] = np.mean(train_times)
            #pprint.pprint(measures)
    pe = time.time()
    print("Completed evaluation in %.2f seconds" % (pe - ps))
