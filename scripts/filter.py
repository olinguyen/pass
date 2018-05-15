import pandas as pd
import numpy as np
from utils import *
from database.query import *

import time


if __name__ == "__main__":
    #tweets_df = pd.read_csv('./data/tweets_with_id.csv', encoding='utf8', sep='\t', index_col=0)
    tweets_df = DataAccess.get_as_dataframe(explode=False)


    with open('./books.txt') as f:
        lines = f.readlines()

    lines = [line.replace('\n', '').lower() for line in lines]
    movies = '|\s(' + '|'.join(lines) + ')'

    with open('./sports.txt') as f:
        lines = f.readlines()

    lines = [line.replace('\n', '').lower() for line in lines]
    outdoors = '|\s(' + '|'.join(lines) + ')'

    with open('./nosleep.txt') as f:
        lines = f.readlines()

    lines = [line.replace('\n', '').lower() for line in lines]
    sleepissues = '|\s(' + '|'.join(lines) + ')'


    sedentary_pattern = r'\s(1984)'
    sedentary_excl = 'job|hire|career|Job|Hiring|Hire|hiring'
    sedentary_pattern = r'^(?=.*(?:%s))(?!.*(?:%s)).*$' % (sedentary_pattern + movies, sedentary_excl)
    print(sedentary_pattern)
    sedentary = tweets_df.text.str.contains(sedentary_pattern) #| tweets_df.hashtags.str.contains(sedentary_pattern)


    pa_pattern = r'\s(playing ball)'
    pa_excl = 'job|hire|career|Job|Hiring|Hire|hiring'
    pa_pattern = r'^(?=.*(?:%s))(?!.*(?:%s)).*$' % (pa_pattern + outdoors, pa_excl)
    print(pa_pattern)
    physical = tweets_df.text.str.contains(pa_pattern) #| tweets_df.hashtags.str.contains(pa_pattern)

    sleeping_pattern = r'\s(cantsleep)'
    sleeping_excl = 'job|hire|career|Job|Hiring|Hire|hiring'
    sleeping_pattern = r'^(?=.*(?:%s))(?!.*(?:%s)).*$' % (sleeping_pattern + sleepissues, sleeping_excl)
    print(sleeping_pattern)
    sleeping = tweets_df.text.str.contains(sleeping_pattern) #| tweets_df.hashtags.str.contains(sleeping_pattern)

    print("Total:", len(tweets_df))
    print("Physical activity:", len(tweets_df.loc[physical == True]))
    print("Sedentary:", len(tweets_df.loc[sedentary == True]))
    print("Sleep:", len(tweets_df.loc[sleeping == True]))

    timestr = time.strftime("%Y%m%d-%H:%M")
    tweets_df.loc[sleeping == True, ['created_at',
      'text',
      'placename']].to_csv('data/tweets_sleeping_%s.csv' % timestr, sep='\t')

    tweets_df.loc[sedentary == True, ['text',
    'placename']].to_csv('data/tweets_sedentary_%s.csv' % timestr, sep='\t')

    tweets_df.loc[physical == True, ['text',
    'placename']].to_csv('data/tweets_physical_%s.csv' % timestr, sep='\t')
