from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from xgboost import XGBClassifier

from feature_extraction.features import *

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

def get_baseline_model():
    return Pipeline([
                ('feature_extraction', TfidfVectorizer(ngram_range=(1,4),
                                                       lowercase=False,
                                                       tokenizer=nltk_tokenizer.tokenize,#LemmaTokenizerSpacy(allowed_postags=['NOUN', 'ADJ', 'ADV']),
                                                       stop_words='english',
                                                       min_df=3,
                                                       max_df=0.9,
                                                       strip_accents='unicode',
                                                       use_idf=True,
                                                       sublinear_tf=True
                                                      )),
                ('feature_selection', SelectFpr(f_classif)), # false positive rate test for feature selection
                ('logistic_regression', GridSearchCV(
                                LogisticRegression(penalty='l2',
                                                   random_state=42),
                                                   param_grid=params))])

def get_ensemble_model():
    return Pipeline([
            ('feature_extraction', get_features()),
            ('feature_selection', SelectFpr(f_classif)), # false positive rate test for feature selection


            ('proba', ProbExtractor([RandomForestClassifier(n_estimators=300,
                                                            max_depth=10,
                                                            min_samples_split=5,
                                                            n_jobs=-1),
                                    ExtraTreesClassifier(n_estimators=300,max_depth=10,
                                                         min_samples_split=10,
                                                         n_jobs=-1),
                                    XGBClassifier(n_estimators=300,
                                                  max_depth=10,
                                                  n_jobs=-1),
                                    LogisticRegression(C=0.1,
                                                       solver='lbfgs',
                                                       penalty='l2',
                                                       n_jobs=-1),
                                    BernoulliNB(alpha=5.0)])),

            ('polynomial', PolynomialFeatures(degree=2)),

            ('logistic_regression', GridSearchCV(
                        LogisticRegression(penalty='l2',
                                           random_state=42),
                                           param_grid=params))])

if __name__ == "__main__":
  print("Pipelines...")
