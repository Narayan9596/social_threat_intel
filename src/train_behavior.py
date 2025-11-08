import os; print("TRAIN BEHAVIOR FILE PATH =>", os.path.abspath(__file__))
print("DEBUG — train_behavior.py is running")
import pandas as pd
from .behavior_model import BehaviorModel

def main():
    print("loading data ...")
    df = pd.read_csv("data/synthetic_twitter.csv")
    print("training behavior model ...")
    model = BehaviorModel()
    model.train(df)
    model.save("behavior_model.joblib")
    print("behavior_model.joblib SAVED")

if __name__ == "__main__":
    main()
