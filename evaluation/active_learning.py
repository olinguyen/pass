import numpy as np
import pandas as pd
import pickle
import dill
import time
from collections import Counter

from database.query import DataAccess
from bson.objectid import ObjectId
from feature_extraction.transformers import *


if __name__ == '__main__':

    ensemble_indicators = ['sleep_ensemble_latest', 'physical_activity_ensemble_latest', 'sedentary_behaviour_ensemble_latest']
    ensemble_models = []

    for indicator in ensemble_indicators:
        with open('./models/%s.pkl' % indicator, 'rb') as f:
            clf = dill.load(f)
            ensemble_models.append((clf, indicator))

    batch_size = 10
    for i in range(10, 100, batch_size):
          print(i/100, (i + batch_size) / 100)

          tstart = time.time()
          tweets_df = DataAccess.sample_control(lower=i / 100, upper= (i + batch_size) / 100)
          #tweets_df = DataAccess.sample_control(lower=0.15, upper=100.0)
          cleaner = TextCleanExtractor()
          tweets_df['clean_text'] = cleaner.transform(tweets_df.text)
          empty = tweets_df.clean_text.apply(lambda x: x == '')
          tweets_df = tweets_df.loc[~empty]
          tend = time.time()

          print("[%.2f/%.2f] Sampled %d tweets in %.2f secs" % (i / 100, (i + batch_size) / 100, len(tweets_df), tend - tstart))


          tstart = time.time()

          for clf, model_name in ensemble_models:
              column_name = model_name + '_predict'
              tweets_df[column_name] = clf.predict_proba(tweets_df.clean_text)[:, 1]

          tend = time.time()

          print("Completed %d predictions in %.3f secs" % (len(tweets_df), tend - tstart))

          sleep_col = 'sleep_ensemble_latest_predict'
          pa_col = 'physical_activity_ensemble_latest_predict'
          sb_col = 'sedentary_behaviour_ensemble_latest_predict'

          sleep = tweets_df[sleep_col] > 0.30
          sb = tweets_df[sb_col] > 0.30
          pa = tweets_df[pa_col] > 0.30

          print("# sleep tweets: %d" % sum(sleep))
          print("# sedentary behavior tweets: %d" % sum(sb))
          print("# physical activity tweets: %d" % sum(pa))

          predicted = tweets_df.loc[sleep | sb | pa]

          outname = './data/test-sampled_ensemble_proba_%d-%dperc.csv' % (i, i + batch_size)
          predicted.to_csv(outname, columns=['created_at',
                          'clean_text',
                          sleep_col,
                          pa_col,
                          sb_col], sep='\t')

          print("Wrote to file:", outname)

          #y_preds = tweets_df.loc[:, [sleep_col, pa_col, sb_col]]
          #DataAccess.write_labels_batch(y_preds)
