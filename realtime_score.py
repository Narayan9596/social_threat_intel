import os; print("RUNNING FILE:", os.path.abspath(__file__))

import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from src.content_model import ContentModel
from src.behavior_model import BehaviorModel
from src.pipeline import FusionPipeline

def get_tweet_data(url):
    # extract tweet ID
    parts = url.split("/")
    tid = parts[-1].split("?")[0]

    scraper = sntwitter.TwitterTweetScraper(tid)
    for i, tweet in enumerate(scraper.get_items()):
        if i == 0:
            user = tweet.user
            now = datetime.datetime.utcnow()
            age_days = (now - user.created).days
            posts_per_day = user.statusesCount / max(age_days,1)

            row = {
                "post_id": tweet.id,
                "user_id": user.id,
                "text": tweet.rawContent,
                "followers": user.followersCount,
                "following": user.friendsCount,
                "tweets_count": user.statusesCount,
                "account_age_days": age_days,
                "retweets": tweet.retweetCount,
                "likes": tweet.likeCount,
                "posts_per_day": posts_per_day
            }
            return row

    return None


if __name__ == "__main__":
    url = input("Enter tweet URL: ")

    data = get_tweet_data(url)
    if data is None:
        print("Tweet not found or invalid URL")
        exit()

    df = pd.DataFrame([data])

    cm = ContentModel.load("content_model.joblib")
    bm = BehaviorModel.load("behavior_model.joblib")
    fusion = FusionPipeline(cm, bm)

    result = fusion.score(df)
    print("\n--- RESULT ---")
    print(result)
