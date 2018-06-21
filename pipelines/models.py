from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, Normalizer, StandardScaler, MaxAbsScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier

from xgboost import XGBClassifier

from feature_extraction.features import *
from feature_extraction.transformers import TextsToSequences, Padder
from nlp.glove import Glove
from models.nn_models import *


params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


def get_baseline_model():
    return Pipeline([
        ('feature_extraction', TfidfVectorizer(ngram_range=(1, 4),
                                               lowercase=False,
                                               tokenizer=nltk_tokenizer.tokenize,
                                               # LemmaTokenizerSpacy(allowed_postags=['NOUN', 'ADJ', 'ADV']),
                                               stop_words='english',
                                               min_df=3,
                                               max_df=0.9,
                                               strip_accents='unicode',
                                               use_idf=True,
                                               sublinear_tf=True
                                               )),
        # false positive rate test for feature selection
        ('feature_selection', SelectFpr(f_classif)),
        ('logistic_regression', GridSearchCV(
            LogisticRegression(penalty='l2',
                                       random_state=42),
            param_grid=params))])


def get_ensemble_model(w2v=None):

    if not w2v:
        glove = Glove.load()
        w2v = glove.get_dict()

    n_jobs = -1
    return Pipeline([

        ('feature_extraction', get_features(w2v)),
        # false positive rate test for feature selection
        ('feature_selection', SelectFpr(f_classif)),
        #('normalize', Normalizer(norm='l2')),

        ('proba', ProbExtractor([RandomForestClassifier(n_estimators=300,
                                                        max_depth=10,
                                                        min_samples_split=5,
                                                        n_jobs=n_jobs),
#                                 ExtraTreesClassifier(n_estimators=300, max_depth=10,
#                                                      min_samples_split=10,
#                                                      n_jobs=n_jobs),
                                 XGBClassifier(n_estimators=300,
                                               max_depth=10,
                                               n_jobs=8),
                                 LogisticRegression(C=0.1,
                                                    solver='lbfgs',
                                                    penalty='l2',
                                                    n_jobs=n_jobs),
                                 BernoulliNB(alpha=5.0)])),

        ('polynomial', PolynomialFeatures(degree=2)),

        ('logistic_regression', GridSearchCV(
            LogisticRegression(penalty='l2',
                               random_state=42),
            param_grid=params))])

def get_basic_model(model, w2v=None):
    if not w2v:
        glove = Glove.load()
        w2v = glove.get_dict()

    n_jobs = -1
    return Pipeline([
        ('feature_extraction', get_features(w2v)),
        # false positive rate test for feature selection
        ('feature_selection', SelectFpr(f_classif)),
        #('normalize', StandardScaler(with_mean=False)),
        #('normalize', MaxAbsScaler()),
        ("model", model)])

def get_keras_model(model):
    vocab_size = 20000
    maxlen = 75

    sequencer = TextsToSequences(num_words=vocab_size)
    padder = Padder(maxlen)

    return Pipeline([
              ("sequencer", sequencer),
              ("padder", padder),
              ("classifier", KerasClassifier(build_fn=model,
                                             epochs=5,
                                             batch_size=256))])

if __name__ == "__main__":
    print("Pipelines...")
