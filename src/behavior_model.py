import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .utils import save_model, load_model

MODEL_NAME = "behavior_model.joblib"

BEHAVIOR_FEATURES = [
    "followers", "following", "tweets_count",
    "account_age_days", "retweets", "likes", "posts_per_day"
]

class BehaviorModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(n_estimators=200, contamination=0.05, random_state=42))
        ])

    def prepare_X(self, df):
        X = df[BEHAVIOR_FEATURES].copy()
        X = X.fillna(0).astype(float)
        for col in ["followers", "tweets_count", "retweets", "likes", "following"]:
            X[col] = np.log1p(X[col])
        return X

    def train(self, df):
        X = self.prepare_X(df)
        self.pipeline.fit(X)
        return self

    def anomaly_score(self, df):
        X = self.prepare_X(df)
        scores = self.pipeline.decision_function(X)
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        anomaly = 1.0 - norm
        return anomaly

    def save(self, name=MODEL_NAME):
        save_model(self.pipeline, name)

    @classmethod
    def load(cls, name=MODEL_NAME):
        pipe = load_model(name)
        obj = cls()
        obj.pipeline = pipe
        return obj
