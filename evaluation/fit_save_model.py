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
import os
import dill


if __name__ == "__main__":

    ps = time.time()

    train_test_data = get_train_test_data()
    ensemble = get_ensemble_model()
    feature_extractor = get_feature_extractor()

    models = [("lr", LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', n_jobs=-1)),
               ("nb", BernoulliNB(alpha=5.0)),
               ("rf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, n_jobs=-1)),
               ("xgb", XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1)),
               ("et", ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_split=10, n_jobs=-1)),
               ("svm", SVC(C=100, gamma=0.0001, probability=True)),
               ("ensemble", ensemble)]

    for Xr_train, y_train, Xr_test, y_test, indicator in train_test_data:
        X_train = feature_extractor.fit_transform(Xr_train, y_train)
        X_test = feature_extractor.transform(Xr_test)

        for name, classifier in models:
            if name == 'ensemble':
                classifier.fit(Xr_train, y_train)
            else:
                classifier.fit(X_train, y_train)

            filename = indicator + '_%s_latest.pkl' % name
            path = os.path.join('./model', filename)
            dill.dump(classifier, open(path, 'wb'))
            print("...wrote to", path)

    pe = time.time()
    print("Completed evaluation in %.2f seconds" % (pe - ps))
