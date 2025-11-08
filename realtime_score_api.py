# realtime_score_api.py
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Print file running (debug)
import os as _os
print("RUNNING FILE:", _os.path.abspath(__file__))

# Load .env from project root (must be next to this file)
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Debug print to ensure keys are loaded (will print None if missing)
print("debug loaded API_KEY:", os.getenv("API_KEY"))

API_KEY = os.getenv("API_KEY")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

if not (API_KEY and API_KEY_SECRET and ACCESS_TOKEN and ACCESS_TOKEN_SECRET):
    print("Missing API credentials in .env. Please add API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET")
    sys.exit(1)

# Third-party imports (require installation)
try:
    import tweepy
except Exception as e:
    print("tweepy is not installed or failed to import:", e)
    print("Install with: py -3.10 -m pip install tweepy python-dotenv")
    sys.exit(1)

try:
    import pandas as pd
    import datetime
except Exception as e:
    print("Missing required packages:", e)
    sys.exit(1)

# local model imports
from src.content_model import ContentModel
from src.behavior_model import BehaviorModel
from src.pipeline import FusionPipeline

def tweet_id_from_url(url_or_id: str) -> str:
    """
    Accepts either a tweet URL or an ID and returns the numeric tweet id.
    """
    # if it's already numeric ID
    if re.fullmatch(r"\d+", url_or_id):
        return url_or_id
    # try parse URL
    m = re.search(r"/status/(\d+)", url_or_id)
    if m:
        return m.group(1)
    raise ValueError("Could not parse tweet ID from input")

def fetch_tweet_and_user(api, tweet_id: str):
    """
    Uses tweepy.API (OAuth1) to fetch tweet and user info.
    Returns a dict with fields expected by your behavior model.
    """
    try:
        status = api.get_status(tweet_id, tweet_mode="extended")
    except tweepy.errors.TweepyException as e:
        raise RuntimeError(f"Twitter API error fetching tweet {tweet_id}: {e}")

    user = status.user

    # account creation date -> days
    created_at = user.created_at  # datetime
    now = datetime.datetime.utcnow()
    account_age_days = max((now - created_at).days, 1)

    posts_per_day = user.statuses_count / account_age_days if account_age_days > 0 else float(user.statuses_count)

    # choose text (full_text if available)
    text = getattr(status, "full_text", None) or getattr(status, "text", "")

    row = {
        "post_id": str(status.id),
        "user_id": str(user.id),
        "text": text,
        "followers": int(user.followers_count),
        "following": int(user.friends_count),
        "tweets_count": int(user.statuses_count),
        "account_age_days": int(account_age_days),
        "retweets": int(getattr(status, "retweet_count", 0)),
        "likes": int(getattr(status, "favorite_count", 0)),
        "posts_per_day": float(posts_per_day),
        "created_at": str(status.created_at)
    }
    return row

def make_api_client():
    """
    Build OAuth1 API client (v1.1 endpoints) for read access using Tweepy.
    """
    auth = tweepy.OAuth1UserHandler(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def score_tweet_from_url(url_or_id: str):
    # 1. build API
    api = make_api_client()
    # 2. parse id
    tid = tweet_id_from_url(url_or_id)
    # 3. fetch tweet => row dict
    row = fetch_tweet_and_user(api, tid)
    df = pd.DataFrame([row])

    # 4. load models (they must exist in models/)
    cm = ContentModel.load("content_model.joblib")
    bm = BehaviorModel.load("behavior_model.joblib")
    fusion = FusionPipeline(cm, bm)

    res = fusion.score(df)
    return res, row

if __name__ == "__main__":
    inp = input("Enter tweet URL or ID: ").strip()
    try:
        res, row = score_tweet_from_url(inp)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)

    print("\n--- TWEET METADATA ---")
    for k,v in row.items():
        print(f"{k}: {v}")

    print("\n--- MODEL OUTPUT ---")
    # convert arrays to simple python floats/ints for nicer display
    content_prob = float(res["content_prob"][0])
    behavior_score = float(res["behavior_score"][0])
    fused = float(res["fused_score"][0])
    verdict = int(res["verdict"][0])

    print(f"content_prob: {content_prob:.4f}")
    print(f"behavior_score: {behavior_score:.4f}")
    print(f"fused_score: {fused:.4f}")
    print(f"verdict: {verdict}   (1 = suspicious)")
