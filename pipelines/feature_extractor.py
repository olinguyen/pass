from feature_extraction.features import get_features, get_glove_w2v
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, f_classif


def get_feature_extractor(w2v=None):
    if not w2v:
        w2v = get_glove_w2v()

    return Pipeline([("feature_extraction", get_features(w2v)),
                     ('feature_selection', SelectFpr(f_classif))
                   ])

