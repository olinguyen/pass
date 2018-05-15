import numpy as np
import pandas as pd
import pickle
import dill
import time
from collections import Counter

from database.query import DataAccess
from bson.objectid import ObjectId
from feature_extraction.transformers import *

if __name__ == "__main__":
    ensemble_indicators = ['sleep_ensemble',
                           'physical_activity_ensemble',
                           'sedentary_behaviour_ensemble']
    ensemble_models = []

    for indicator in ensemble_indicators:
        with open('./model/%s.pkl' % indicator, 'rb') as f:
            clf = dill.load(f)
            ensemble_models.append((clf, indicator))

    tstart = time.time()

    tweets_df = DataAccess.sample_control(lower=0.1, upper=0.20)

    cleaner = TextCleanExtractor()
    tweets_df['clean_text'] = cleaner.transform(tweets_df.text)
    empty = tweets_df.clean_text.apply(lambda x: x == '')
    tweets_df = tweets_df.loc[~empty]
    tend = time.time()

    print("Sampled %d tweets in %.2f secs" % (len(tweets_df), tend - tstart))
    tweets_df.head()

    tstart = time.time()

    for clf, model_name in ensemble_models:
        column_name = model_name + '_predict'
        tweets_df[column_name] = clf.predict_proba(tweets_df.clean_text)[:, 1]

    tend = time.time()

    print("Completed %d predictions in %.3f secs" % (len(tweets_df), tend - tstart))

    tweets_df.head()

    sleep = tweets_df['sleep_ensemble_predict'] > 0.3
    sb = tweets_df['sedentary_behaviour_ensemble_predict'] > 0.35
    pa = tweets_df['physical_activity_ensemble_predict'] > 0.3

    print("# sleep tweets: %d" % sum(sleep))
    print("# sedentary behavior tweets: %d" % sum(sb))
    print("# physical activity tweets: %d" % sum(pa))

    predicted = tweets_df.loc[sleep | sb | pa]

    predicted.to_csv('./data/sampled_ensemble_proba_10-20perc.csv', columns=['created_at',
                              'clean_text',
                              'sleep_ensemble_predict',
                              'physical_activity_ensemble_predict',
                              'sedentary_behaviour_ensemble_predict'], sep='\t')


    y_preds = tweets_df.loc[:, ['sleep_ensemble_predict',
                      'sedentary_behaviour_ensemble_predict',
                      'physical_activity_ensemble_predict']]

    DataAccess.write_labels_batch(y_preds)
