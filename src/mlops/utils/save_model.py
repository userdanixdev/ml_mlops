import joblib
from pathlib import Path

def save_model(model,model_name):
    model_path = Path("models") / f"{model_name}.joblib"
    model_path.parent.mkdir(parents=True,exist_ok=True)

    joblib.dump(model,model_path)

    print(f"\nModelo salvo em: {model_path}")