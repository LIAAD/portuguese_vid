import pymongo
from tqdm import tqdm
import json
from pathlib import Path
import os
import pandas as pd
import dotenv
from datasets import Dataset, Features, Value, ClassLabel
import re
import demoji

dotenv.load_dotenv(dotenv.find_dotenv())

client = pymongo.MongoClient("localhost", 27017)

db = client.data

tweets = db.tweets
users = db.users

CURRENT_PATH = Path(__file__).parent

filtered_locations = json.load(open(os.path.join(
    CURRENT_PATH, 'filtered_locations.json'), 'r', encoding='utf-8'))


def validate_user(user):

    if user.get('_id') is None:
        return False

    if user.get('most_common_language') != 'pt':
        return False

    if (location := user.get('location')) is None:
        return False

    elif location.lower() not in filtered_locations:
        return False

    return True


def beautify_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = demoji.replace(text, '')

    return text


def get_tweets(users):
    print(f"Getting tweets from {len(users)} users")

    tweets = db.tweets.find({'user': {'$in': users}})

    results = []

    for tweet in tqdm(tweets):

        if tweet.get('lang') != 'pt':
            continue

        results.append(beautify_text(tweet.get('full_text')))

    print(f"Found {len(results)} tweets")

    return results


def process_users():
    skip = 0

    tweets = []

    for limit in tqdm(range(1000, 200000, 1000)):
        valid_users = []

        for user in tqdm(users.find().skip(skip).limit(limit)):
            if validate_user(user):
                valid_users.append(user['_id'])

        tweets += get_tweets(valid_users)

        skip += limit

    return tweets


tweets = process_users()

df = pd.DataFrame(tweets, columns=['text'])

df['label'] = 'pt-PT'

dataset = Dataset.from_pandas(df, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
    'portuguese_presidential_elections_tweets_lid', token=os.getenv('HF_TOKEN')
)
