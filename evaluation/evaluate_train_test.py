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

    ps = time.time()

    train_test_data = get_train_test_data()
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

    for Xr_train, y_train, Xr_test, y_test, indicator in train_test_data:
        X_train = feature_extractor.fit_transform(Xr_train, y_train)
        X_test = feature_extractor.transform(Xr_test)

        results[indicator] = {}
        for name, classifier in models:
            results[indicator][name] = {}

            cv = StratifiedKFold(n_splits=5, random_state=42)
            scores = []
            conf_mat = np.zeros((2, 2))      # Binary classification
            false_pos = set()
            false_neg = set()
            train_times = []

            for dev_i, val_i in cv.split(X_train, y_train):
                clf = deepcopy(classifier)
                X_dev, X_val = X_train[dev_i], X_train[val_i]
                y_dev, y_val = y_train[dev_i], y_train[val_i]
                ts = time.time()

                clf.fit(X_dev, y_dev)
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

            classifier.fit(X_train, y_train)
            y_scores_test = classifier.predict_proba(X_test)
            results[indicator][name]['test_roc_auc'] = roc_auc_score(y_test, y_scores_test[:, 1])

            print("\n[%s][%s] Mean score: %0.2f (+/- %0.2f)" % (indicator, name, np.mean(scores), np.std(scores) * 2))

            if name == 'ensemble':
                filename = indicator + '_ensemble_latest.pkl'
                path = os.path.join('./model', filename)
                dill.dump(clf, open(path, 'wb'))
                print("...wrote to", path)

            #conf_mat /= 5
            #print("Mean CM: \n", conf_mat)
            #print("\nMean classification measures: \n")
            measures = class_report(conf_mat)
            for metric in measures:
                results[indicator][name][metric] = measures[metric]

            results[indicator][name]['mean_roc_auc'] = np.mean(scores)
            results[indicator][name]['std_roc_auc'] = np.std(scores) * 2
            results[indicator][name]['train_time'] = np.mean(train_times)

    pe = time.time()
    print("Completed evaluation in %.2f seconds" % (pe - ps))
