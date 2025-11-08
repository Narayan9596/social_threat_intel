import numpy as np
import pandas as pd
from .content_model import ContentModel
from .behavior_model import BehaviorModel

class FusionPipeline:
    def __init__(self, content_model: ContentModel, behavior_model: BehaviorModel,
                 content_weight=0.6, behavior_weight=0.4, threshold=0.5):
        self.content_model = content_model
        self.behavior_model = behavior_model
        self.content_weight = content_weight
        self.behavior_weight = behavior_weight
        self.threshold = threshold

    def score(self, df):
        texts = df['text'].fillna("").tolist()
        content_probs = self.content_model.predict_proba(texts)
        behavior_scores = self.behavior_model.anomaly_score(df)
        fused = (self.content_weight * content_probs) + (self.behavior_weight * behavior_scores)

        return {
            "content_prob": content_probs,
            "behavior_score": behavior_scores,
            "fused_score": fused,
            "verdict": (fused >= self.threshold).astype(int)
        }
