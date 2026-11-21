
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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


def generate_optimization_report(
    best_params,
    best_score,
    accuracy,
    report_dict,
    matrix
):
    report_path = Path(
        "reports/optimization/randomized_search_report.md"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Relatório de Otimização - RandomizedSearchCV

## Configuração da busca

- Modelo: Random Forest
- Método: RandomizedSearchCV
- Número de combinações: 20
- Validação cruzada: 5 folds
- Métrica de seleção: F1-macro
- Random state: 42

## Melhores hiperparâmetros

```python
{best_params}
```

## Resultados

- Melhor F1-macro na validação cruzada: {best_score:.4f}
- Accuracy no conjunto de teste: {accuracy:.4f}

## Matriz de confusão

```text
{matrix}
```

## Métricas de classificação

```text
{pd.DataFrame(report_dict).T.to_string()}
```
"""

    report_path.write_text(report, encoding="utf-8")

    print(f"\nRelatório de otimização salvo em: {report_path}")


def optimize_random_forest():

    # Carregamento dos dados
    data = load_dataset()

    # Separação das features e target
    X, y = separate_features_target(data)

    # Verificação do pré-processamento
    check_transformations(X)

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Modelo base
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    # Hiperparâmetros para otimização
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    # Busca aleatória
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    print("\nIniciando RandomizedSearchCV...")

    randomized_search.fit(X_train, y_train)

    # Melhores resultados
    best_model = randomized_search.best_estimator_

    print("\nMelhores hiperparâmetros:")
    print(randomized_search.best_params_)

    print("\nMelhor resultado na validação cruzada:")
    print(f"{randomized_search.best_score_:.4f}")

    # Avaliação final no conjunto de teste
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=["malignant", "benign"],
        zero_division=0,
        output_dict=True
    )

    print("\nAvaliação do modelo otimizado")
    print("*" * 40)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nMatriz de confusão:")
    print(matrix)

    print("\nRelatório de classificação:")
    print(pd.DataFrame(report_dict).T)

    # Salvamento do modelo
    save_model(best_model, "model_3_random_forest_optimized")

    # Salvamento das métricas para comparação
    save_metrics(
        model_name="Random Forest Optimized",
        accuracy=accuracy,
        classification_metrics=report_dict
    )

    # Salvamento dos resultados da otimização
    results_path = Path(
        "reports/optimization/randomized_search_results.csv"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(randomized_search.cv_results_).to_csv(
        results_path,
        index=False
    )

    print(f"\nResultados da busca salvos em: {results_path}")

    # Geração do relatório da otimização
    generate_optimization_report(
        best_params=randomized_search.best_params_,
        best_score=randomized_search.best_score_,
        accuracy=accuracy,
        report_dict=report_dict,
        matrix=matrix
    )


if __name__ == "__main__":
    optimize_random_forest()
