from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ExplodingRecordJoiner(BaseEstimator, TransformerMixin):
    """
    ExplodingRecordJoiner
    ~~~~~~~~~~~~~~~~~~~~~
    ExplodingRecordJoiner is a Transformer for Pipeline Objects
    Usage:
        The reason we use this is because of the fact that
        using DataFrams is better than using JSON parsing.
        However, the data coming in is nested JSON so this exploder
        allows use to select a `col` that is one level nested dictionary
        (taken from json) and selects the `subcol` and joins
        it to the original DataFrame.
    """

    def __init__(self, **kwargs):
        self.cols = kwargs

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        # Extract column of dicts then apply from_records,
        # Match indicies then select the `subcols` we want,
        # Join with existing DataFrame.
        if "geo" in self.cols:
            for row in X.loc[X.geo.isnull(), "geo"].index:
                X.at[row, 'geo'] = {"coordinates": []}

        for col, subcol in self.cols.items():
            new_cols = ["{}.{}".format(col, c) for c in subcol]
            sub = pd.DataFrame.from_records(X[col], index=X.index)[subcol]
            sub.columns = new_cols
            X = X.join(sub)
            del X[col]

        X['latitude'] = X['geo.coordinates'].apply(lambda x: x[0] if x else None)
        X['longitude'] = X['geo.coordinates'].apply(lambda x: x[1] if x else None)
        del X['geo.coordinates']
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def __repr__(self):
        st = [k + "=" + str(v) for k, v in self.cols.items()]
        return "ExplodingRecordJoiner({})".format(", ".join(st))

ready_made_exploder = ExplodingRecordJoiner(
    user = [
        'created_at',
        'favourites_count',
        'followers_count',
        'friends_count',
        'statuses_count',
        'verified'],
    place = ['full_name', 'country'],
    geo = ['coordinates'],
    entities = ['hashtags']
)
