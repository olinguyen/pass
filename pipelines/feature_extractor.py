from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, f_classif

from feature_extraction.features import get_features
from nlp.glove import Glove

def get_feature_extractor(w2v=None):
    if not w2v:
        glove = Glove.load()
        w2v = glove.get_dict()

    return Pipeline([("feature_extraction", get_features(w2v)),
                     ('feature_selection', SelectFpr(f_classif))
                   ])

