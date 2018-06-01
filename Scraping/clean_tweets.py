from pprint import pprint

import pandas as pd
import csv
import re

def clean_tweets(fname):
    tweets = pd.read_csv("{}.csv".format(fname), error_bad_lines=False, quoting=csv.QUOTE_NONE)
    try:
        tweets = tweets.drop(tweets[tweets.is_retweet == "true"].index)
    except Exception:
        pass
    tweets = tweets.text
    for i in range(tweets.shape[0]):
        try:
            tweets.loc[i] = (re.sub('(http[^ ]*)', '<hyperlink>', tweets.loc[i])).encode('utf8').decode()
            if i % 100 == 0:
                print(i)
                print(tweets.loc[i])
        except KeyError:
            print("whoops"+str(i))
    tweets.to_csv('../Data/clean {}.csv'.format(fname), index=False)
# pprint(trtweets)

clean_tweets("trump")
clean_tweets("cnn")

