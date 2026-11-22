
import ast
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlops.tracking.mlflow_config import configure_mlflow


# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OPTIMIZATION_DIR = REPORTS_DIR / "optimization"

METRICS_PATH = REPORTS_DIR / "model_metrics.csv"


# Configuração dos modelos baseline
BASELINE_MODELS = {
    "Random Forest Baseline": {
        "model_name": "Random Forest",
        "model_path": MODELS_DIR / "model_1_random_forest.joblib",
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": 1,
        },
    },
    "Logistic Regression Baseline": {
        "model_name": "Logistic Regression",
        "model_path": MODELS_DIR / "model_2_logistic_regression.joblib",
        "params": {
            "scaler": "StandardScaler",
            "classifier": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 421,
        },
    },
}


# Configuração dos modelos otimizados
OPTIMIZED_MODELS = {
    "RandomizedSearchCV": {
        "model_name": "Random Forest Optimized",
        "model_path": MODELS_DIR / "model_3_random_forest_optimized.joblib",
        "results_path": OPTIMIZATION_DIR / "randomized_search_results.csv",
        "report_path": OPTIMIZATION_DIR / "randomized_search_report.md",
        "plots": [
            "randomized_search_best_metrics.png",
            "randomized_search_parameters.png",
            "randomized_search_scores.png",
        ],
    },
    "HalvingGridSearchCV": {
        "model_name": "Random Forest Halving Optimized",
        "model_path": MODELS_DIR / "model_4_random_forest_halving_optimized.joblib",
        "results_path": OPTIMIZATION_DIR / "halving_search_results.csv",
        "report_path": OPTIMIZATION_DIR / "halving_search_report.md",
        "plots": [
            "halving_best_model_evaluation.png",
            "halving_candidates_per_iteration.png",
            "halving_resources_per_iteration.png",
        ],
    },
}


def get_test_metrics(model_name):
    """Recupera as métricas de teste do modelo."""

    metrics_df = pd.read_csv(METRICS_PATH)

    model_metrics = metrics_df.loc[
        metrics_df["model"] == model_name
    ]

    if model_metrics.empty:
        raise ValueError(
            f"Métricas não encontradas para: {model_name}"
        )

    metrics = model_metrics.iloc[0]

    logged_metrics = {}

    for metric_name, value in metrics.items():

        if metric_name == "model" or metric_name.endswith("_support"):
            continue

        metric_key = (
            f"test_{metric_name}"
            .replace(" ", "_")
            .replace("-", "_")
        )

        logged_metrics[metric_key] = float(value)

    return logged_metrics


def get_best_results(results_path):
    """Recupera os melhores hiperparâmetros da otimização."""

    results = pd.read_csv(results_path)

    best_row = results.loc[
        results["mean_test_score"].idxmax()
    ]

    best_params = ast.literal_eval(best_row["params"])

    return best_params, best_row


def log_model_artifacts(model, model_path):
    """Registra o modelo no formato MLflow e o Joblib original."""

    mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ],
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="joblib_backup",
    )


def log_test_metrics(model_name):
    """Registra as métricas de avaliação no conjunto de teste."""

    metrics = get_test_metrics(model_name)

    mlflow.log_metrics(metrics)


def register_baseline(run_name, config):
    """Registra um modelo baseline no MLflow."""

    model_path = config["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}"
        )

    model = joblib.load(model_path)

    with mlflow.start_run(run_name=run_name) as run:

        # Hiperparâmetros utilizados no baseline
        mlflow.log_params({
            key: str(value)
            for key, value in config["params"].items()
        })

        # Identificação do experimento
        mlflow.set_tags({
            "model_stage": "baseline",
            "dataset": "Breast Cancer Wisconsin",
            "estimator": run_name,
        })

        # Métricas de teste
        log_test_metrics(config["model_name"])

        # Modelo e arquivo original
        log_model_artifacts(model, model_path)

        # CSV geral de métricas
        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="reports",
        )

        print(f"\nBaseline registrado: {run_name}")
        print(f"Run ID: {run.info.run_id}")


def register_optimization(method, config):
    """Registra um modelo otimizado no MLflow."""

    model_path = config["model_path"]
    results_path = config["results_path"]
    report_path = config["report_path"]

    required_files = [
        model_path,
        results_path,
        report_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {file_path}"
            )

    # Recupera os melhores resultados da busca
    best_params, best_row = get_best_results(results_path)

    # Carrega o modelo já treinado
    model = joblib.load(model_path)

    with mlflow.start_run(run_name=method) as run:

        # Melhores hiperparâmetros
        mlflow.log_params({
            key: str(value)
            for key, value in best_params.items()
        })

        # Identificação da otimização
        mlflow.set_tags({
            "model_stage": "optimized",
            "optimization_method": method,
            "dataset": "Breast Cancer Wisconsin",
            "estimator": "RandomForestClassifier",
        })

        # Métricas da validação cruzada
        mlflow.log_metric(
            "cv_f1_macro",
            float(best_row["mean_test_score"]),
        )

        mlflow.log_metric(
            "cv_f1_macro_std",
            float(best_row["std_test_score"]),
        )

        # Métricas de teste
        log_test_metrics(config["model_name"])

        # Modelo e arquivo original
        log_model_artifacts(model, model_path)

        # Resultados e relatório da otimização
        mlflow.log_artifact(
            str(results_path),
            artifact_path="reports",
        )

        mlflow.log_artifact(
            str(report_path),
            artifact_path="reports",
        )

        # Gráficos da otimização
        for plot_name in config["plots"]:

            plot_path = OPTIMIZATION_DIR / plot_name

            if plot_path.exists():
                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="plots",
                )

        # CSV geral de métricas
        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="reports",
        )

        print(f"\nOtimização registrada: {method}")
        print(f"Run ID: {run.info.run_id}")
        print(
            f"F1 Macro CV: "
            f"{best_row['mean_test_score']:.4f}"
        )


def main():

    # Configura o MLflow Tracking
    configure_mlflow()

    print("\n========== MODELOS BASELINE ==========")

    for run_name, config in BASELINE_MODELS.items():
        register_baseline(run_name, config)

    print("\n========== MODELOS OTIMIZADOS ==========")

    for method, config in OPTIMIZED_MODELS.items():
        register_optimization(method, config)

    print("\nTodos os modelos foram registrados no MLflow Tracking!")


if __name__ == "__main__":
    main()
