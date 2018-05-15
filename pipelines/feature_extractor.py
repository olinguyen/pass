from feature_extraction.features import get_features
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, f_classif


def get_feature_extractor():
    return Pipeline([("feature_extraction", get_features()),
                     ('feature_selection', SelectFpr(f_classif))])

