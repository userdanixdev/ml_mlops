
from pathlib import Path

import pandas as pd

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from mlops.data.load_dataset import load_dataset
from mlops.preprocessing.pre_process import (
    separate_features_target,
    check_transformations,
    train_test_split_data
)

from mlops.utils.save_model import save_model
from mlops.utils.save_metrics import save_metrics


def generate_halving_report(
    search,
    accuracy,
    classification_metrics,
    confusion_mat
):
    output_dir = Path("reports/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Informações das iterações
    iterations = []

    for i, (candidates, resources) in enumerate(
        zip(search.n_candidates_, search.n_resources_)
    ):
        iterations.append(
            f"| {i + 1} | {candidates} | {resources} |"
        )

    iterations_table = "\n".join(iterations)

    # Métricas de classificação
    metrics_table = []

    for class_name in ["malignant", "benign", "macro avg", "weighted avg"]:
        metrics = classification_metrics[class_name]

        metrics_table.append(
            f"| {class_name} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1-score']:.4f} | "
            f"{int(metrics['support'])} |"
        )

    metrics_table = "\n".join(metrics_table)

    # Matriz de confusão
    confusion_table = f"""
| | Predito: Malignant | Predito: Benign |
|---|---:|---:|
| Real: Malignant | {confusion_mat[0][0]} | {confusion_mat[0][1]} |
| Real: Benign | {confusion_mat[1][0]} | {confusion_mat[1][1]} |
"""

    # Melhores hiperparâmetros
    best_params = "\n".join(
        f"- **{param}:** `{value}`"
        for param, value in search.best_params_.items()
    )

    report = f"""# Relatório de Otimização - HalvingGridSearchCV

## 1. Visão geral

O HalvingGridSearchCV foi utilizado para otimizar o modelo Random Forest
por meio do algoritmo Successive Halving.

- **Modelo:** Random Forest Classifier
- **Estratégia:** HalvingGridSearchCV
- **Validação cruzada:** 5 folds
- **Métrica de otimização:** F1 Macro
- **Fator de eliminação:** 3

## 2. Melhores hiperparâmetros

{best_params}

## 3. Resultado da otimização

- **Melhor F1 Macro (Cross-validation):** {search.best_score_:.4f}
- **Acurácia no conjunto de teste:** {accuracy:.4f}
- **Quantidade de iterações:** {search.n_iterations_}

## 4. Evolução das iterações

| Iteração | Candidatos avaliados | Recursos utilizados |
|---|---:|---:|
{iterations_table}

## 5. Avaliação do melhor modelo

### Métricas de classificação

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
{metrics_table}

### Matriz de confusão

{confusion_table}

## 6. Considerações finais

O HalvingGridSearchCV realizou a otimização do Random Forest
por meio da eliminação progressiva de combinações de hiperparâmetros.

O melhor estimador encontrado foi avaliado no conjunto de teste,
permitindo analisar seu desempenho em dados não utilizados durante
o treinamento e a otimização.

Os resultados detalhados das avaliações estão disponíveis em:

`reports/optimization/halving_search_results.csv`

O modelo otimizado foi armazenado em:

`models/model_4_random_forest_halving_optimized.joblib`
"""

    report_path = output_dir / "halving_search_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Relatório Markdown salvo em: {report_path}")

def optimize_random_forest_halving():

    # 1. Carregamento dos dados
    data = load_dataset()

    # 2. Separação das features e target
    X, y = separate_features_target(data)

    # 3. Verificação dos dados
    check_transformations(X)

    # 4. Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 5. Definição do modelo
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    # 6. Espaço de busca
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    # 7. Configuração do HalvingGridSearchCV
    halving_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        factor=3,
        resource="n_samples",
        min_resources="smallest",
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    # 8. Execução da otimização
    halving_search.fit(X_train, y_train)

    # 9. Melhores resultados
    print("\nMelhores hiperparâmetros:")
    print(halving_search.best_params_)

    print("\nMelhor F1 Macro:")
    print(halving_search.best_score_)

    print("\nQuantidade de iterações:")
    print(halving_search.n_iterations_)

    print("\nCandidatos por iteração:")
    print(halving_search.n_candidates_)

    print("\nRecursos por iteração:")
    print(halving_search.n_resources_)

    # 10. Recuperação do melhor modelo
    best_model = halving_search.best_estimator_

    # 11. Avaliação com dados de teste
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["malignant", "benign"],
        zero_division=0,
        output_dict=True
    )

    matrix = confusion_matrix(y_test, y_pred)

    print("\nAcurácia:", accuracy)

    print("\nRelatório de classificação:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["malignant", "benign"],
        zero_division=0
    ))

    print("\nMatriz de confusão:")
    print(matrix)

    # 12. Salvamento do modelo
    save_model(
        best_model,
        "model_4_random_forest_halving_optimized"
    )

    # 13. Salvamento das métricas
    save_metrics(
        model_name="Random Forest Halving Optimized",
        accuracy=accuracy,
        classification_metrics=report_dict
    )

    # 14. Salvamento dos resultados da otimização
    results = pd.DataFrame(halving_search.cv_results_)

    results = results.sort_values(
        by="rank_test_score"
    )

    output_dir = Path("reports/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "halving_search_results.csv",
        index=False
    )

    print("\nResultados da otimização salvos em:")
    print(output_dir / "halving_search_results.csv")
    
    # 15. Geração do relatório Markdown
    generate_halving_report(
        search=halving_search,
        accuracy=accuracy,
        classification_metrics=report_dict,
        confusion_mat=matrix
    )


if __name__ == "__main__":
    optimize_random_forest_halving()
