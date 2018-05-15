import pandas as pd
from feature_extraction.transformers import TextCleanExtractor


def get_train_test_data(clean=True):
    cleaner = TextCleanExtractor()

    train_sleep = pd.read_csv('./data/train_sleep.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_sleep'])
    test_sleep = pd.read_csv('./data/test_sleep.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_sleep'])
    train_sleep = train_sleep.dropna().loc[train_sleep.label_sleep != -1].reset_index(drop=True)
    test_sleep = test_sleep.dropna().loc[test_sleep.label_sleep != -1].reset_index(drop=True)
    train_sleep['clean_text'] = cleaner.transform(train_sleep.clean_text)
    test_sleep['clean_text'] = cleaner.transform(test_sleep.clean_text)

    train_pa = pd.read_csv('./data/train_pa.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_pa'])
    test_pa = pd.read_csv('./data/test_pa.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_pa'])
    train_pa = train_pa.dropna().loc[train_pa.label_pa!= -1].reset_index(drop=True)
    test_pa = test_pa.dropna().loc[test_pa.label_pa!= -1].reset_index(drop=True)
    train_pa['clean_text'] = cleaner.transform(train_pa.clean_text)
    test_pa['clean_text'] = cleaner.transform(test_pa.clean_text)

    train_sb = pd.read_csv('./data/train_sb.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_sb'])
    test_sb = pd.read_csv('./data/test_sb.tsv', sep='\t', usecols=['_id', 'clean_text', 'label_sb'])
    train_sb = train_sb.dropna().loc[train_sb.label_sb!= -1].reset_index(drop=True)
    test_sb = test_sb.dropna().loc[test_sb.label_sb!= -1].reset_index(drop=True)
    train_sb['clean_text'] = cleaner.transform(train_sb.clean_text)
    test_sb['clean_text'] = cleaner.transform(test_sb.clean_text)

    data = [(train_sleep['clean_text'], train_sleep['label_sleep'], test_sleep['clean_text'], test_sleep['label_sleep'], 'sleep'),
            (train_sb['clean_text'], train_sb['label_sb'], test_sb['clean_text'], test_sb['label_sb'], 'sedentary_behaviour'),
            (train_pa['clean_text'], train_pa['label_pa'], test_pa['clean_text'], test_pa['label_pa'], 'physical_activity')]

    return data


def get_labeled_data(clean=True):
    cleaner = TextCleanExtractor()

    sleep_tweets_df = pd.read_csv('data/sleep_labeled_dec22.csv')
    sleep_tweets_df = sleep_tweets_df.loc[:, ['text', 'hashtags', 'placename', 'first_person_sleep_problem']]
    sleep_tweets_df['clean_text'] = cleaner.transform(sleep_tweets_df.text)
    n_samples = len(sleep_tweets_df)
    n_positives = sum(sleep_tweets_df.first_person_sleep_problem == True)
    print("[Sleep] True labels: %d/%d (%.3f%%)" % (n_positives, n_samples, n_positives / n_samples * 100.0))

    sb_tweets_df = pd.read_csv('data/sedentary_labeled_april.csv')
    sb_tweets_df = sb_tweets_df.loc[:, ['text', 'hashtags', 'placename', 'first_person_sedentary_behavior']]
    sb_tweets_df['clean_text'] = cleaner.transform(sb_tweets_df.text)

    n_samples = len(sb_tweets_df)
    n_positives = sum(sb_tweets_df.first_person_sedentary_behavior == True)
    print("[Sedentary Behavior] True labels: %d/%d (%.3f%%)" % (n_positives, n_samples, n_positives / n_samples * 100.0))

    pa_tweets_df = pd.read_csv('data/pa_labeled_dec22.csv')
    pa_tweets_df = pa_tweets_df.loc[:, ['text', 'hashtags', 'placename', 'first_person_physical_activity']]
    pa_tweets_df['clean_text'] = cleaner.transform(pa_tweets_df.text)

    n_samples = len(pa_tweets_df)
    n_positives = sum(pa_tweets_df.first_person_physical_activity == True)
    print("[Physical Activity] True labels: %d/%d (%.3f%%)" % (n_positives, n_samples, n_positives / n_samples * 100.0))
    pa_tweets_df.head()

    tweets_df = pd.concat((pa_tweets_df, sleep_tweets_df, sb_tweets_df), ignore_index=True)
    tweets_df.head()

    X = tweets_df.clean_text
    y = tweets_df[['first_person_sleep_problem',
                   'first_person_sedentary_behavior',
                   'first_person_physical_activity']]

    X_sleep = sleep_tweets_df.clean_text
    y_sleep = sleep_tweets_df.first_person_sleep_problem

    X_sb = sb_tweets_df.clean_text
    y_sb = sb_tweets_df.first_person_sedentary_behavior

    X_pa = pa_tweets_df.clean_text
    y_pa = pa_tweets_df.first_person_physical_activity


    data = [(X_sleep, y_sleep, 'sleep'),
            (X_sb, y_sb, 'sedentary_behaviour'),
            (X_pa, y_pa, 'physical_activity')]

    return data

if __name__ == "__main__":
    get_labeled_data()
