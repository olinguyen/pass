import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


from database.utils import get_train_test_data
from pipelines.feature_extractor import get_feature_extractor

import time

parameters = {"n_estimators": [100, 200, 300, 400],
              "max_depth": [7, 8, 9, 10]}


if __name__ == "__main__":
    print("Running grid search for logistic regression on parameters: %s" %
      parameters)

    train_test_data = get_train_test_data()
    feature_extractor = get_feature_extractor()

    for Xr_train, y_train, _, _, indicator in train_test_data:
        def classification_report_with_auc_score(y_true, y_pred):
            y_trues.extend(y_true)
            y_preds.extend(y_pred)
            #print(classification_report(y_true, y_pred))
            return roc_auc_score(y_true, y_pred) # return accuracy score

        ts = time.time()
        xgb = XGBClassifier()

        X_feats = feature_extractor.fit_transform(Xr_train, y_train)

        cv = GridSearchCV(xgb,
                  param_grid=parameters,
                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                  refit=True,
                  scoring='roc_auc',
                  n_jobs=-1)

        cv.fit(X_feats, y_train)
        te = time.time()

        y_trues = []
        y_preds = []

        print("Trained in %.2f" % (te - ts))
        print()
        print(indicator, cv.best_params_, cv.best_score_)

        print("Detailed classification report across 5 folds:")
        print()

        cross_val_score(cv, X_feats, y_train,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring=make_scorer(classification_report_with_auc_score), n_jobs=8)

        print(classification_report(y_trues, y_preds))
        print()
