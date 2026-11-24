import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from mlops.tracking.mlflow_config import configure_mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORTS_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
REPORT_PATH = REPORTS_DIR / "model_selection_report.md"

load_dotenv()
REGISTERED_RUNS = {
    "Random Forest Baseline": os.getenv("MLFLOW_RUN_RANDOM_FOREST"),
    "Logistic Regression Baseline": os.getenv("MLFLOW_RUN_LOGISTIC_REGRESSION"),
    "RandomizedSearchCV": os.getenv("MLFLOW_RUN_RANDOMIZED_SEARCH"),
    "HalvingGridSearchCV": os.getenv("MLFLOW_RUN_HALVING_SEARCH"),
}

# Identificar o modelo pelo Run ID.
def get_model_name(run_id):
    for model_name, registered_run_id in REGISTERED_RUNS.items():
        if registered_run_id == run_id:
            return model_name

    return "Modelo desconhecido"
# Assim, conseguimos identificar a origem de cada versão sem deixar Run IDs fixos no código.

# Formatar métricas para exibição no relatório.
def format_metric(value):
    if value is None:
        return "N/D"

    return f"{value:.4f}"

REGISTERED_MODEL_NAME = "breast-cancer-classifier"

# Gerar relatório de avaliação dos modelos.
def generate_evaluation_report(evaluated_candidates,best_model):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = [
        "# Model Evaluation & Selection",
        "",
        f"**Data da avaliação:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        "## Relatório Modelo de Seleção:",
        "",
         "Avaliar e comparar os modelos registrados no MLflow Model Registry, "
        "utilizando métricas do conjunto de teste e validação cruzada "
        "quando disponíveis, para fundamentar a seleção do candidato.",
        "",
        "Os modelos foram ordenados considerando os seguintes critérios, "
        "em ordem de prioridade:",
        "",
        "1. **Recall Malignant:** priorizar a identificação de casos malignos.",
        "2. **F1 Macro:** avaliar o equilíbrio entre as classes.",
        "3. **Accuracy:** considerar o desempenho geral de classificação.",
        "",
        "A ordenação utiliza critérios hierárquicos, sem ponderação "
        "ou combinação matemática entre as métricas.",
        "## Resultados comparativos",
        "",
        "| Versão | Modelo | Accuracy | F1 Macro | Recall Malignant | CV F1 Macro |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, candidate in evaluated_candidates.iterrows():
        report.append(
            f"| {candidate['rank']} "
            f"| {candidate['version']} "
            f"| {candidate['model']} "
            f"| {format_metric(candidate['test_malignant_recall'])} "
            f"| {format_metric(candidate['test_f1_macro'])} "
            f"| {format_metric(candidate['test_accuracy'])} "
            f"| {format_metric(candidate['cv_f1_macro'])} |"
        )

    report.extend([
        "",
        "## 4. Modelo selecionado",
        "",
        f"**Modelo:** {best_model['model']}",
        "",
        f"**Versão no Registry:** {best_model['version']}",
        "",
        "### Métricas do candidato selecionado",
        "",
        f"- Recall Malignant: {format_metric(best_model['test_malignant_recall'])}",
        f"- F1 Macro: {format_metric(best_model['test_f1_macro'])}",
        f"- Accuracy: {format_metric(best_model['test_accuracy'])}",
        f"- CV F1 Macro: {format_metric(best_model['cv_f1_macro'])}",
        "",
        "### Justificativa técnica",
        "",
        "O candidato foi selecionado por apresentar o maior Recall Malignant "
        "entre os modelos avaliados. Os critérios subsequentes, F1 Macro "
        "e Accuracy, foram utilizados como desempate, conforme a ordem "
        "de prioridade definida.",
        "",
        "## Observações e limitações",
        "",
        "- As métricas de teste foram recuperadas dos Runs registrados "
        "no MLflow Tracking.",
        "- CV F1 Macro está disponível somente para os modelos otimizados.",
        "- A ausência de CV F1 Macro nos modelos baseline não foi "
        "interpretada como desempenho nulo.",
        "- A seleção representa uma comparação entre os candidatos "
        "avaliados, não uma garantia de desempenho em dados futuros.",
        "- A seleção automática não substitui a validação de requisitos "
        "clínicos, análise de erros e avaliação em dados independentes.",
        "",
        
    ])

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"\nRelatório gerado: {REPORT_PATH}")

# A função abaixo consulta o Registry e retorna as versões existentes.
def load_registered_versions():
    configure_mlflow()

    client = MlflowClient()
    versions = client.search_model_versions(
        filter_string=f"name='{REGISTERED_MODEL_NAME}'"
    )
    if not versions:
        raise ValueError(
            f"Nenhuma versão encontrada para {REGISTERED_MODEL_NAME}."
        )        
    return versions

# Recuperar as métricas das Runs:

def get_model_metrics(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)

    return run.data.metrics

# Exibir métricas dos modelos registrados.
def display_model_metrics(versions):
    for version in versions:
        metrics = get_model_metrics(version.run_id)

        print(f"\nVersão: {version.version}")
        print(f"Accuracy: {metrics.get('test_accuracy')}")
        print(f"F1 Macro: {metrics.get('test_macro_avg_f1_score')}")
        print(f"Recall Malignant: {metrics.get('test_malignant_recall')}")
        print(f"CV F1 Macro: {metrics.get('cv_f1_macro', 'Não disponível')}")

# Exibir versões registradas no Registry.
def display_registered_versions(versions):
    print(f"\nTotal de versões encontradas: {len(versions)}")

    for version in versions:
        print(
            f"Modelo: {version.name} | "
            f"Versão: {version.version} | "
            f"Run ID: {version.run_id}"
        )

# Construir tabela comparativa dos modelos.
def build_comparison_table(versions):
    results = []

    for version in versions:
        metrics = get_model_metrics(version.run_id)

        results.append({
            "version": version.version,
            "model": get_model_name(version.run_id),
            "run_id": version.run_id,
            "test_accuracy": metrics.get("test_accuracy"),
            "test_f1_macro": metrics.get("test_macro_avg_f1_score"),
            "test_malignant_recall": metrics.get("test_malignant_recall"),
            "cv_f1_macro": metrics.get("cv_f1_macro"),
        })

    return pd.DataFrame(results)

# Avaliar e ordenar os candidatos pelos critérios técnicos.
def evaluate_candidates(comparison_table):
    required_metrics = [
        "test_malignant_recall",
        "test_f1_macro",
        "test_accuracy",
    ]

    if comparison_table.empty:
        raise ValueError("Não existem modelos para avaliar.")

    missing_metrics = comparison_table[required_metrics].isnull().any(axis=1)

    if missing_metrics.any():
        invalid_models = comparison_table.loc[missing_metrics, "model"].tolist()

        raise ValueError(
            f"Existem modelos sem métricas obrigatórias: {invalid_models}"
        )

    evaluated = comparison_table.sort_values(
        by=[
            "test_malignant_recall",
            "test_f1_macro",
            "test_accuracy",
        ],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    evaluated.insert(
        0,
        "rank",
        range(1, len(evaluated) + 1)
    )

    evaluated["evaluation"] = evaluated["rank"].apply(
        lambda rank: "Candidato prioritário" if rank == 1
        else "Candidato avaliado"
    )

    return evaluated

# Selecionar o melhor candidato.
def select_best_model(evaluated_candidates):
    if evaluated_candidates.empty:
        raise ValueError("Não existem candidatos avaliados.")

    best_model = evaluated_candidates.iloc[0]

    return best_model

def main():
    versions = load_registered_versions()

    display_registered_versions(versions)
    display_model_metrics(versions)

    comparison_table = build_comparison_table(versions)
    evaluated_candidates = evaluate_candidates(comparison_table)
    best_model = select_best_model(evaluated_candidates)

    print("\n===== CLASSIFICAÇÃO DOS CANDIDATOS =====")
    print(
        evaluated_candidates[
            [
                "rank",
                "model",
                "test_malignant_recall",
                "test_f1_macro",
                "test_accuracy",
                "cv_f1_macro",
                "evaluation",
            ]
        ].to_string(index=False)
    )

    print("\n===== MODELO SELECIONADO =====")
    print(f"Modelo: {best_model['model']}")
    print(f"Versão: {best_model['version']}")
    print(f"Recall Malignant: {best_model['test_malignant_recall']:.4f}")
    print(f"F1 Macro: {best_model['test_f1_macro']:.4f}")
    print(f"Accuracy: {best_model['test_accuracy']:.4f}")

    generate_evaluation_report(evaluated_candidates,best_model)

    
if __name__ == "__main__":
    main()
    
