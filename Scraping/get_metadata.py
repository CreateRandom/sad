import tweepy
import json
import math
import glob
import csv
import zipfile
import CONFIG
import zlib
from tweepy import TweepError
from time import sleep

# CHANGE THIS TO THE USER YOU WANT
user = 'CNN'

keys = CONFIG.keys

auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
auth.set_access_token(keys['access_token'], keys['access_token_secret'])
api = tweepy.API(auth)
user = user.lower()
output_file = '{}.json'.format(user)
output_file_short = '{}_short.json'.format(user)
compression = zipfile.ZIP_DEFLATED

try:
		with open('all_ids.json') as f:
				ids = json.load(f)
except FileNotFoundError:
		print("no such file yet")
		ids = []

print('total ids: {}'.format(len(ids)))

all_data = []
start = 0
end = 100
limit = len(ids)
i = math.ceil(limit / 100)

for go in range(i):
    print('currently getting {} - {}'.format(start, end))
    sleep(6)  # needed to prevent hitting API rate limit
    id_batch = ids[start:end]
    start += 100
    end += 100
    tweets = api.statuses_lookup(id_batch)
    for tweet in tweets:
        all_data.append(dict(tweet._json))

print('metadata collection complete')
print('creating master json file')
with open(output_file, 'w') as outfile:
    json.dump(all_data, outfile)

print('creating zipped master json file')
zf = zipfile.ZipFile('{}.zip'.format(user), mode='w')
zf.write(output_file, compress_type=compression)
zf.close()

results = []

def is_retweet(entry):
    return 'retweeted_status' in entry.keys()

def get_source(entry):
    if '<' in entry["source"]:
        return entry["source"].split('>')[1].split('<')[0]
    else:
        return entry["source"]

with open(output_file) as json_data:
    data = json.load(json_data)
    for entry in data:
        t = {
            "created_at": entry["created_at"],
            "text": entry["text"],
            "in_reply_to_screen_name": entry["in_reply_to_screen_name"],
            "retweet_count": entry["retweet_count"],
            "favorite_count": entry["favorite_count"],
            "source": get_source(entry),
            "id_str": entry["id_str"],
            "is_retweet": is_retweet(entry)
        }
        results.append(t)

print('creating minimized json master file')
with open(output_file_short, 'w') as outfile:
    json.dump(results, outfile)
with open(output_file_short) as master_file:
    data = json.load(master_file)
    fields = ["text", "id_str"]
    print('creating CSV version of minimized json master file')
    f = csv.writer(open('{}.csv'.format(user), 'w'))
    f.writerow(fields)
    for x in data:
        try:
            f.writerow([x['text'], x["id_str"]])
        except Exception as e:
            print(e)
