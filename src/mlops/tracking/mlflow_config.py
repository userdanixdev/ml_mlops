
import mlflow
from pathlib import Path

# Diretório de armazenamento do MLflow
MLFLOW_DIR = Path("mlruns")
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Banco SQLite para armazenar os metadados
TRACKING_URI = "sqlite:///mlflow_project_cancer.db"

# Nome do experimento
EXPERIMENT_NAME = "Breast Cancer Wisconsin"


def configure_mlflow():

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow Tracking configurado com sucesso!")
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Experimento: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    configure_mlflow()

# MLFLOW_DIR: diretório destinado aos artefatos dos experimentos. 
# Ainda precisamos configurar explicitamente seu uso.
# TRACKING_URI: informa que os metadados serão armazenados no banco SQLite mlflow_project_cancer.db.
# EXPERIMENT_NAME: identifica o projeto dentro da interface do MLflow.
# configure_mlflow(): configura a URI e seleciona ou cria o experimento.