# Model Evaluation & Selection

**Data da avaliação:** 23/11/2026 23:07:44

## Relatório Modelo de Seleção:

Avaliar e comparar os modelos registrados no MLflow Model Registry, utilizando métricas do conjunto de teste e validação cruzada quando disponíveis, para fundamentar a seleção do candidato.

Os modelos foram ordenados considerando os seguintes critérios, em ordem de prioridade:

1. **Recall Malignant:** priorizar a identificação de casos malignos.
2. **F1 Macro:** avaliar o equilíbrio entre as classes.
3. **Accuracy:** considerar o desempenho geral de classificação.

A ordenação utiliza critérios hierárquicos, sem ponderação ou combinação matemática entre as métricas.
## Resultados comparativos

| Versão | Modelo | Accuracy | F1 Macro | Recall Malignant | CV F1 Macro |
|---|---|---:|---:|---:|---:|
| 1 | 2 | Logistic Regression Baseline | 0.9762 | 0.9812 | 0.9825 | nan |
| 2 | 4 | HalvingGridSearchCV | 0.9286 | 0.9526 | 0.9561 | 0.9464 |
| 3 | 3 | RandomizedSearchCV | 0.9286 | 0.9526 | 0.9561 | 0.9554 |
| 4 | 1 | Random Forest Baseline | 0.9286 | 0.9435 | 0.9474 | nan |

## Modelo selecionado

**Modelo:** Logistic Regression Baseline

**Versão no Registry:** 2

### Métricas do candidato selecionado

- Recall Malignant: 0.9762
- F1 Macro: 0.9812
- Accuracy: 0.9825
- CV F1 Macro: nan

### Justificativa técnica

O candidato foi selecionado por apresentar o maior Recall Malignant entre os modelos avaliados. Os critérios subsequentes, F1 Macro e Accuracy, foram utilizados como desempate, conforme a ordem de prioridade definida.

## Observações e limitações

- As métricas de teste foram recuperadas dos Runs registrados no MLflow Tracking.
- CV F1 Macro está disponível somente para os modelos otimizados.
- A ausência de CV F1 Macro nos modelos baseline não foi interpretada como desempenho nulo.
- A seleção representa uma comparação entre os candidatos avaliados, não uma garantia de desempenho em dados futuros.
- A seleção automática não substitui a validação de requisitos clínicos, análise de erros e avaliação em dados independentes.
