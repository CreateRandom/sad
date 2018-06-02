from pprint import pprint

import pandas as pd
import csv
import re
import nltk


RANDOM_SEED = 42


def tag_tweets(fname):
    """ Loads the clean tweets and adds tags

    """
    tweets = pd.read_csv("../Data/clean_{}.csv".format(fname),
                         error_bad_lines=False, index_col=None,
                         quoting=csv.QUOTE_NONE, engine='python',
                         header=None, sep='\n')
    tweet_counter = tweets.shape[0]
    for i in range(tweet_counter):
        tweets.loc[i][0] = re.sub(' [ ]*', '  ', tweets.loc[i][0])
        tweets.loc[i][0] = tweets.loc[i][0].split("  ")
        try:
            tweets.loc[i][0] = nltk.pos_tag(tweets.loc[i][0])
        except IndexError:
            tweets.loc[i][0] = []
        if i % 100 == 0:
            print(i)
            print(tweets.loc[i][0])
    tweets.to_csv('../Data/tagged_{}.csv'.format(fname), index=False,
                  quoting=csv.QUOTE_NONE, escapechar='\\', sep="\t")

tag_tweets("trump")
tag_tweets("cnn")
