import joblib
from pathlib import Path

def save_model(model):
    model_path = Path("models/model_1_random_forest.joblib")
    model_path.parent.mkdir(parents=True,exist_ok=True)

    joblib.dump(model,model_path)