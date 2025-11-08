import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

def save_model(obj, name):
    path = MODEL_DIR / name
    joblib.dump(obj, path)
    return str(path)

def load_model(name):
    path = MODEL_DIR / name
    return joblib.load(path)
