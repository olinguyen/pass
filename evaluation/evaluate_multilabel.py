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
from nlp.glove import Glove


from copy import deepcopy
import numpy as np
import pandas as pd
import pprint
import time


if __name__ == "__main__":

    ps = time.time()

    Xr_train, y_train, Xr_test, y_test = get_train_test_data(merge=True)

    glove = Glove.load()
    w2v = glove.get_dict()

    ensemble = get_ensemble_model(w2v)
    ensemble.steps = ensemble.steps[2:]
    feature_extractor = get_feature_extractor(w2v)

    models = [("lr", OneVsRestClassifier(LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', n_jobs=-1))),
              ("nb", OneVsRestClassifier(BernoulliNB(alpha=5.0))),
              ("rf", OneVsRestClassifier(RandomForestClassifier(n_estimators=300,
                                                                max_depth=10,
                                                                min_samples_split=5,
                                                                n_jobs=-1))),
              ("xgb", OneVsRestClassifier(XGBClassifier(n_estimators=150,
                                                        max_depth=8,
                                                        n_jobs=8))),
              ("et", OneVsRestClassifier(ExtraTreesClassifier(n_estimators=300,
                                                              max_depth=10,
                                                              min_samples_split=10,
                                                              n_jobs=-1))),
              ("ensemble", OneVsRestClassifier(ensemble)),
               #("svm", SVC(C=100, gamma=0.0001, probability=True)),
             ]

    results = {}

    X_train = feature_extractor.fit_transform(Xr_train, y_train['label_pa'])
    X_test = feature_extractor.transform(Xr_test)

    for name, classifier in models:
        print(name)
        results[name] = {}

        cv = StratifiedKFold(n_splits=5, random_state=42)

        train_times = []
        predict_times = []

        ts = time.time()
        classifier.fit(X_train, y_train)
        te = time.time()
        train_times.append(te - ts)

        y_scores_test = classifier.predict_proba(X_test)
        ts = time.time()
        y_pred_test = classifier.predict(X_test)
        te = time.time()
        predict_times.append(te - ts)

        measures, tpr, fpr = class_report_multilabel(y_test, y_scores_test)
        for metric in measures:
            results[name][metric] = measures[metric]

        results[name]['train_time'] = np.mean(train_times)
        results[name]['predict_time'] = np.mean(predict_times)

    pe = time.time()
    print("Completed evaluation in %.2f seconds" % (pe - ps))
