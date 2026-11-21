# Relatório de Otimização - HalvingGridSearchCV

## 1. Visão geral

O HalvingGridSearchCV foi utilizado para otimizar o modelo Random Forest
por meio do algoritmo Successive Halving.

- **Modelo:** Random Forest Classifier
- **Estratégia:** HalvingGridSearchCV
- **Validação cruzada:** 5 folds
- **Métrica de otimização:** F1 Macro
- **Fator de eliminação:** 3

## 2. Melhores hiperparâmetros

- **max_depth:** `20`
- **max_features:** `sqrt`
- **min_samples_leaf:** `1`
- **min_samples_split:** `5`
- **n_estimators:** `500`

## 3. Resultado da otimização

- **Melhor F1 Macro (Cross-validation):** 0.9464
- **Acurácia no conjunto de teste:** 0.9561
- **Quantidade de iterações:** 3

## 4. Evolução das iterações

| Iteração | Candidatos avaliados | Recursos utilizados |
|---|---:|---:|
| 1 | 72 | 20 |
| 2 | 24 | 60 |
| 3 | 8 | 180 |

## 5. Avaliação do melhor modelo

### Métricas de classificação

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| malignant | 0.9512 | 0.9286 | 0.9398 | 42 |
| benign | 0.9589 | 0.9722 | 0.9655 | 72 |
| macro avg | 0.9551 | 0.9504 | 0.9526 | 114 |
| weighted avg | 0.9561 | 0.9561 | 0.9560 | 114 |

### Matriz de confusão


| | Predito: Malignant | Predito: Benign |
|---|---:|---:|
| Real: Malignant | 39 | 3 |
| Real: Benign | 2 | 70 |


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
