import os; print("RUNNING FILE:", os.path.abspath(__file__))

from src.content_model import ContentModel
from src.behavior_model import BehaviorModel
from src.pipeline import FusionPipeline
import pandas as pd

cm = ContentModel.load("content_model.joblib")
bm = BehaviorModel.load("behavior_model.joblib")

fusion = FusionPipeline(cm, bm)

newrow = {
  "post_id":"abcd1",
  "user_id":"u1",
  "text":"BREAKING - You won't believe this exclusive claim about celebrity!",
  "followers":10,
  "following":5,
  "tweets_count":2,
  "account_age_days":3,
  "retweets":0,
  "likes":0,
  "posts_per_day":0.02
}

df = pd.DataFrame([newrow])
result = fusion.score(df)

print(result)
