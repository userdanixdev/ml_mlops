
# Avaliação do Modelo Logistic Regression

Esta etapa tem como objetivo avaliar o desempenho inicial do modelo `Logistic Regression`, utilizando o dataset `Breast Cancer Wisconsin.`

> A avaliação foi realizada com o conjunto de teste, separado durante a etapa de pré-processamento dos dados.

*O modelo Logistic Regression alcançou **98,25% de acurácia** no conjunto de teste.*

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9825 |
| Total de amostras | 114 |
| Previsões corretas | 112 |
| Previsões incorretas | 2 |

A matriz de confusão obtida foi:

```text
[[41  1]
 [ 1 71]]
```

Considerando a ordem das classes (`malignant` e `benign`):

| Classe real | Predito malignant | Predito benign |
|---|---:|---:|
| Malignant | 41 | 1 |
| Benign | 1 | 71 |

**O modelo classificou corretamente 41 dos 42 casos malignos e 71 dos 72 casos benignos.**

Foram registrados dois erros de classificação:

- Um caso maligno classificado como benigno.
- Um caso benigno classificado como maligno.

## Relatório de classificação

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.98 | 0.98 | 0.98 | 42 |
| Benign | 0.99 | 0.99 | 0.99 | 72 |
| Macro avg | 0.98 | 0.98 | 0.98 | 114 |
| Weighted avg | 0.98 | 0.98 | 0.98 | 114 |

### Interpretação das métricas

- **Precision:** indica a proporção de previsões de determinada classe que estavam corretas.
- **Recall:** representa a capacidade do modelo de identificar corretamente os registros reais de determinada classe.
- **F1-score:** representa a média harmônica entre Precision e Recall.
- **Support:** indica a quantidade de registros reais de cada classe no conjunto de teste.

*O recall de 0.98 para a classe `malignant` indica que o modelo identificou corretamente 41 dos 42 casos malignos.*

## Conclusão

O modelo Logistic Regression apresentou um desempenho inicial satisfatório, alcançando 98,25% de acurácia e F1-score de 0.98 para a classe `malignant`.

Os resultados demonstram boa capacidade de classificação neste conjunto de teste. Entretanto, houve um caso maligno classificado incorretamente como benigno, o que merece atenção na análise das métricas.

## 6. Próximas etapas

1. Consolidar os resultados do Random Forest e da Regressão Logística.
2. Comparar as métricas dos dois modelos.
3. Otimizar os hiperparâmetros utilizando RandomizedSearchCV e HalvingGridSearchCV.
4. Integrar o MLflow Tracking para registrar e comparar os experimentos.
