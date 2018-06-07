import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from database.utils import get_labeled_data
from pipelines.feature_extractor import get_feature_extractor

import time

parameters = {"alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0],
              "binarize": [0.0, 0.25, 0.50, 0.75, 1.0]}


if __name__ == "__main__":
    print("Running grid search for logistic regression on parameters: %s" %
          parameters)

    data = get_labeled_data()
    feature_extractor = get_feature_extractor()

    for X, y, indicator in data:

        def classification_report_with_auc_score(y_true, y_pred):
            y_trues.extend(y_true)
            y_preds.extend(y_pred)
            #print(classification_report(y_true, y_pred))
            return roc_auc_score(y_true, y_pred)  # return accuracy score

        ts = time.time()
        bnb = BernoulliNB()

        X_feats = feature_extractor.fit_transform(X, y)

        cv = GridSearchCV(
            bnb,
            param_grid=parameters,
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42),
            refit=True,
            scoring='roc_auc',
            n_jobs=-1)

        cv.fit(X_feats, y)
        te = time.time()

        y_trues = []
        y_preds = []

        print("Trained in %.2f" % (te - ts))
        print()
        print(indicator, cv.best_params_, cv.best_score_)

        print("Detailed classification report across 5 folds:")
        print()

        cross_val_score(
            cv,
            X_feats,
            y,
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42),
            scoring=make_scorer(classification_report_with_auc_score))

        print(classification_report(y_trues, y_preds))
        print()
