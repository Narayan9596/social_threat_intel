from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .utils import save_model, load_model

MODEL_NAME = "content_model.joblib"

class ContentModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def train(self, df):
        X = df["text"].fillna("")
        y = df["label"].astype(int)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, texts):
        probs = self.pipeline.predict_proba(texts)
        return probs[:,1]

    def save(self, name=MODEL_NAME):
        save_model(self.pipeline, name)

    @classmethod
    def load(cls, name=MODEL_NAME):
        pipe = load_model(name)
        obj = cls()
        obj.pipeline = pipe
        return obj
