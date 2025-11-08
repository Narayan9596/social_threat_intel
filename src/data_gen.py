import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def rand_date(days=90):
    base = datetime.now()
    delta = timedelta(days=random.randint(0, days))
    return (base - delta).isoformat()

def make_row(label):
    text = fake.sentence()
    followers = random.randint(10,20000)
    following = random.randint(1,2000)
    tweets = random.randint(1,10000)
    age = random.randint(1,3000)
    retweets = random.randint(0,200)
    likes = random.randint(0,300)
    posts_per_day = random.uniform(0.01,5.0)

    return {
        "post_id": fake.uuid4(),
        "user_id": fake.uuid4(),
        "text": text,
        "followers": followers,
        "following": following,
        "tweets_count": tweets,
        "account_age_days": age,
        "retweets": retweets,
        "likes": likes,
        "posts_per_day": posts_per_day,
        "created_at": rand_date(),
        "label": label
    }

def generate():
    rows=[]
    for i in range(1000):
        label = random.choice([0,1])
        rows.append(make_row(label))
    df=pd.DataFrame(rows)
    df.to_csv("data/synthetic_twitter.csv",index=False)
    print("created data/synthetic_twitter.csv")

if __name__=="__main__":
    generate()
