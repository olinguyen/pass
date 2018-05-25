from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

from database.utils import get_labeled_data, get_train_test_data
from pipelines.feature_extractor import get_feature_extractor
from pipelines.models import get_ensemble_model
from evaluation.metrics import class_report


from copy import deepcopy
import numpy as np
import pandas as pd
import pprint
import time


if __name__ == "__main__":
    ensemble = get_ensemble_model()
    ensemble.steps = ensemble.steps[2:]
    feature_extractor = get_feature_extractor()

    Xr_train, yr_train, Xr_test, yr_test = get_train_test_data(merge=True)

    models = [("lr", LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', n_jobs=-1)),
              ("nb", BernoulliNB(alpha=5.0)),
              ("rf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, n_jobs=-1)),
              ("xgb", XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1)),
              ("et", ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=10, n_jobs=-1)),
              ("svm", SVC(C=100, gamma=0.0001, probability=True)),
              ("ensemble", ensemble),
             ]

    results = {}

    for label in cols_target:
        print('... Processing {}'.format(label))
        y_train = yr_train[label]
        y_test = yr_test[label]

        X_train = feature_extractor.fit_transform(Xr_train, y_train)
        X_test = feature_extractor.transform(Xr_test)
        print(label, X_train.shape)

        results[label] = {}
        for name, classifier in models:
            results[label][name] = {}

            cv = StratifiedKFold(n_splits=5, random_state=42)
            scores = []
            conf_mat = np.zeros((2, 2))      # Binary classification
            false_pos = set()
            false_neg = set()
            train_times = []
            predict_times = []

            for dev_i, val_i in cv.split(X_train, y_train):
                clf = deepcopy(classifier)
                X_dev, X_val = X_train[dev_i], X_train[val_i]
                y_dev, y_val = y_train[dev_i], y_train[val_i]
                ts = time.time()
                clf.fit(X_dev, y_dev)
                te = time.time()

                train_times.append(te - ts)

                ts = time.time()
                y_pprobs = clf.predict_proba(X_val)       # Predicted probabilities
                te = time.time()

                predict_times.append(te - ts)

                y_plabs = np.squeeze(clf.predict(X_val))  # Predicted class labels

                scores.append(roc_auc_score(y_val, y_pprobs[:, 1]))

                confusion = confusion_matrix(y_val, y_plabs)
                conf_mat += confusion

                # Collect indices of false positive and negatives
                fp_i = np.where((y_plabs==1) & (y_val==0))[0]
                fn_i = np.where((y_plabs==0) & (y_val==1))[0]
                false_pos.update(val_i[fp_i])
                false_neg.update(val_i[fn_i])

            classifier.fit(X_train, y_train)
            y_scores_test = classifier.predict_proba(X_test)
            mean_roc_auc = np.mean(scores)
            test_roc_auc = roc_auc_score(y_test, y_scores_test[:, 1])
            results[label][name]['test_roc_auc'] = test_roc_auc

            print("\n[%s][%s] 5-fold CV Mean score: %0.2f (+/- %0.2f)" % (label, name, mean_roc_auc, np.std(scores) * 2))
            print("\n[%s][%s] Test score: %0.2f" % (label, name, test_roc_auc))

            #conf_mat /= 5
            #print("Mean CM: \n", conf_mat)
            #print("\nMean classification measures: \n")
            measures = class_report(conf_mat)
            for metric in measures:
                results[label][name][metric] = measures[metric]

            results[label][name]['mean_roc_auc'] = mean_roc_auc
            results[label][name]['std_roc_auc'] = mean_roc_auc * 2
            results[label][name]['train_time'] = np.mean(train_times)
            results[label][name]['predict_time'] = np.mean(predict_times)
