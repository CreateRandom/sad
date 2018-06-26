import pandas as pd
import csv
import re


RANDOM_SEED = 42


def clean_tweets(fname, with_quoting=False):
    """ Loads the data and cleans the tweets

    Removes hyperlinks and usernames
    takes out retweets
    Makes format equal for both files
    Splits in test and train tweets
    saves as clean_<name>.csv and test_<name>.csv"""
    if with_quoting:
        tweets = pd.read_csv("{}.csv".format(fname), error_bad_lines=False,
                             encoding='cp1252')
    else:
        tweets = pd.read_csv("{}.csv".format(fname), error_bad_lines=False,
                             quoting=csv.QUOTE_NONE)
    tweets.dropna(subset=['text'], inplace=True)
    tweet_counter = tweets.shape[0]-1
    try:
        tweets = tweets.drop(tweets[tweets.is_retweet == "true"].index)
    except Exception:
        pass
    tweets = tweets.text
    for i in range(tweet_counter):
        try:
            tweets.loc[i] = re.sub('(http[^ ]*)', 'httpHYPERLINK', tweets.loc[
                i]).encode('utf8').decode()
            tweets.loc[i] = re.sub('@[a-zA-Z0-9_]*', '@USERNAME',
                                   tweets.loc[i])
            tweets.loc[i] = re.sub('[\n\r]', '\t', tweets.loc[i])
            tweets.loc[i] = re.sub('&amp;', '&', tweets.loc[i])
            if i % 100 == 0:
                print(i)
                print(tweets.loc[i])
        except KeyError:  # Already dropped as retweet
            print("whoops"+str(i))
        except TypeError:  # Bad tweet, drop it or else wont save the csv.
            tweets.drop(i)
            print("whoops"+str(i)+" {}".format(tweets.loc[i]))

    tweets, test_tweets = holdout(tweets)
    tweets.to_csv('../Data/clean_{}.csv'.format(fname), index=False,
                  quoting=csv.QUOTE_NONE, escapechar="\\", sep="\n")
    test_tweets.to_csv('../Data/test_{}.csv'.format(fname), index=False,
                       quoting=csv.QUOTE_NONE, escapechar="\\", sep="\n")


def holdout(tweets):
    """ Creates a test set of tweets which will be kept out of the set.

    :param tweets: the dataframe of tweets.
    :returns tweets: tweets to train on
    :returns test_tweets: tweets that will be used in the survey"""
    test_tweets = tweets.sample(300, random_state=RANDOM_SEED)
    tweets = tweets.drop(test_tweets.index)
    return tweets.to_frame(), test_tweets.to_frame()


clean_tweets("trump")
clean_tweets("cnn", True)
