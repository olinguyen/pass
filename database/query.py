import pymongo
import json
import csv

import numpy as np
import pandas as pd
from bson.objectid import ObjectId

from database.helper import ready_made_exploder

client = pymongo.MongoClient()
col = client['tweets']['december']


def union(*dtts):
    result = {}
    for d in dtts:
        result.update(d)
    return result


def project(*attributes, prefix=None):
    temp = prefix + ".{}" if prefix else "{}"
    return {temp.format(attr): 1 for attr in attributes}


def exists(label, e=True):
    return {label: {"$exists": e}}


class Projections:
    """
    This services as the initial feature set used to project data
    from MongoDB, the code there should also serve as a reference
    """
    text = project("text")
    labels = project("labels")
    time = project("created_at")
    control = project("control")
    random = project("random_number")
    language = project("lang")
    retweeted = project("retweeted")

    hashtags = project("hashtags",
                       prefix="entities")
    geo = project("coordinates",
                  prefix="geo")
    place = project(
        "country",
        "city",
        "full_name",
        "province",
        prefix="place")
    user = project(
        "friends_count",
        "followers_count",
        "statuses_count",
        "favourites_count",
        "created_at",
        "verified",
        prefix="user"
    )

    #all = union(text, hashtags, place, geo, place)
    all = union(
        text,
        time,
        user,
        labels,
        language,
        hashtags,
        geo,
        place,
        retweeted)


class Queries:
    """
    This serves as a collection of different queries we do on the collection from MongoDB.
    """
    X = exists("labels")
    no_label = exists("labels", e=False)

    @classmethod
    def sample(cls, lower=0.0, upper=1.0):
        """
        Each item in the collection has a random number,
        this way we can deterministically sample the collection.
        For all elements:
            `random_number in (lower, upper)`
        :param lower:
        :param upper:
        :return: query object for MongoDB
        """
        return {"random_number":
                {
                    "$gt": lower,
                    "$lt": upper
                }
                }


class DataAccess:

    @classmethod
    def sample_control(cls, lower=0, upper=.01, explode=True):
        df = cls.to_df(
            col.find(
                Queries.sample(
                    lower,
                    upper),
                projection=Projections.all))
        if explode:
            return ready_made_exploder.fit_transform(df)
        else:
            return df

    @classmethod
    def to_df(cls, cursor):
        return pd.DataFrame(list(cursor)).set_index("_id")

    @classmethod
    def get_as_dataframe(
            cls,
            find=Queries.no_label,
            projection=Projections.all,
            explode=True):
        df = cls.to_df(col.find(find, projection))
        if explode:
            return ready_made_exploder.fit_transform(df)
        else:
            return df

    @classmethod
    def get_not_labeled(cls):
        return cls.to_df(
            col.find(
                Queries.no_label,
                Projections.mechanical_turk))

    @classmethod
    def write_labels(cls, series, column_name):
        for _id, label in series.to_dict().items():
            if isinstance(label, np.int64):
                label = label.item()
            col.find_one_and_update({"_id": ObjectId(_id)}, {
                                    "$set": {column_name: label}})

    @classmethod
    def write_labels_batch(cls, dataframe):
        for _id, dict_labels in dataframe.T.to_dict().items():
            for key, value in dict_labels.items():
                if isinstance(value, np.int64):
                    dict_labels[key] = value.item()
            col.find_one_and_update({"_id": ObjectId(_id)}, {
                                    "$set": dict_labels})

    @classmethod
    def count_withlabels(cls):
        return col.find(exists("labels")).count()
