import pandas as pd
from .content_model import ContentModel

def main():
    df = pd.read_csv("data/synthetic_twitter.csv")
    model = ContentModel()
    model.train(df)
    model.save("content_model.joblib")
    print("content_model.joblib SAVED")

if __name__ == "__main__":
    main()
