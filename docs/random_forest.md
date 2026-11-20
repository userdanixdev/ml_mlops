# Baseline - Random Forest

Após a conclusão das etapas de carregamento e pré-processamento dos dados, foi realizado o primeiro treinamento de um modelo de Machine Learning.

O objetivo desta etapa é estabelecer um **baseline**, ou seja, um modelo de referência que permita avaliar o desempenho inicial do projeto e comparar experimentos futuros. Para isso, foi utilizado o algoritmo **Random Forest Classifier**, disponibilizado pela biblioteca Scikit-learn.

O Random Forest é um algoritmo de aprendizado supervisionado baseado em um conjunto de árvores de decisão. Durante o treinamento, são construídas diversas árvores, cujas previsões são combinadas para determinar a classificação final. Essa abordagem permite obter um modelo robusto e adequado para problemas de classificação, como a identificação das classes presentes no dataset Breast Cancer Wisconsin.

> Além disso, o Random Forest não exige padronização das escalas das variáveis, diferentemente de outros algoritmos, como Regressão Logística e SVM.

O modelo foi instanciado utilizando os seguintes parâmetros:

| Parâmetro | Valor | Descrição |
|---|---:|---|
| `n_estimators` | 300 | Quantidade de árvores utilizadas na floresta. |
| `max_depth` | None | Permite que as árvores cresçam sem uma profundidade máxima definida. |
| `min_samples_split` | 2 | Número mínimo de amostras necessário para dividir um nó. |
| `min_samples_leaf` | 1 | Número mínimo de amostras permitido em um nó folha. |
| `random_state` | 42 | Define a semente aleatória para permitir a reprodução dos resultados. |
| `n_jobs` | -1 | Utiliza todos os processadores disponíveis para execução paralela. |

> A configuração inicial foi mantida para estabelecer uma referência de desempenho antes de realizar possíveis ajustes nos hiperparâmetros.

O treinamento foi implementado no módulo `mlops/training/train.py`, utilizando a função `train_model()`.

```python
def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model
```

## Avaliação do modelo

A avaliação foi realizada utilizando os dados de teste, que não participaram do treinamento.

Para isso, foi utilizado o método `predict()` para gerar as previsões e as métricas disponibilizadas pelo módulo `sklearn.metrics`.

As métricas utilizadas foram:

- **Accuracy:** representa a proporção de previsões corretas em relação ao total de observações.
- **Precision:** indica a proporção de previsões positivas que realmente pertencem à classe correspondente.
- **Recall:** mede a proporção de observações reais de uma classe que foram identificadas corretamente.
- **F1-score:** representa a média harmônica entre Precision e Recall.
- **Confusion Matrix:** apresenta a distribuição dos acertos e erros de classificação entre as classes.

Após a execução do treinamento e da avaliação, foram obtidos os seguintes resultados:

**Dimensões dos conjuntos**

| Conjunto | Registros | Features |
|---|---:|---:|
| X_train | 455 | 30 |
| X_test | 114 | 30 |
| y_train | 455 | - |
| y_test | 114 | - |

**Accuracy**

O modelo alcançou uma acurácia de **94,74%**, acertando 108 das 114 observações presentes no conjunto de teste.

**Matriz de confusão**

```text
[[39  3]
 [ 3 69]]
```

Considerando a ordem das classes `malignant` e `benign`:

| Classe real | Previsto maligno | Previsto benigno |
|---|---:|---:|
| Malignant | 39 | 3 |
| Benign | 3 | 69 |

A matriz demonstra que:

- 39 observações malignas foram classificadas corretamente.
- 3 observações malignas foram classificadas incorretamente como benignas.
- 69 observações benignas foram classificadas corretamente.
- 3 observações benignas foram classificadas incorretamente como malignas.

**Relatório de classificação**

| Classe | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.93 | 0.93 | 0.93 | 42 |
| Benign | 0.96 | 0.96 | 0.96 | 72 |
| Macro avg | 0.94 | 0.94 | 0.94 | 114 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 114 |

O relatório apresenta resultados satisfatórios para ambas as classes. Entretanto, merece destaque o Recall de 0.93 para a classe `malignant`, indicando que aproximadamente 93% dos casos malignos presentes no conjunto de teste foram identificados corretamente.

Em problemas relacionados à classificação de tumores, é especialmente importante acompanhar os casos malignos classificados incorretamente como benignos.

> Esses resultados são experimentais e não representam uma validação clínica do modelo.

Para manter a separação de responsabilidades, o treinamento e a avaliação foram implementados em um módulo próprio.

A função `train_model()` é responsável pela criação e pelo treinamento do modelo, enquanto `evaluate_model()` realiza as previsões e calcula as métricas.

O arquivo `main.py` atua como ponto de entrada e orquestra as etapas:

1. Carregamento do dataset.
2. Separação entre features e target.
3. Verificação dos dados.
4. Divisão entre treinamento e teste.
5. Treinamento do modelo.
6. Avaliação dos resultados.

*Essa organização facilita a manutenção e permite incorporar novas funcionalidades sem concentrar todas as responsabilidades em um único arquivo.*

## Conclusão

O primeiro treinamento foi concluído com sucesso, estabelecendo um baseline de **94,74% de Accuracy** utilizando o algoritmo Random Forest.

Esse resultado representa uma referência inicial para futuras experimentações, permitindo comparar diferentes configurações de hiperparâmetros e outros algoritmos de classificação.

Como próxima etapa, será implementada a persistência do modelo treinado, permitindo armazená-lo e recuperá-lo posteriormente sem a necessidade de executar novamente o treinamento.

Após essa etapa, será realizada a integração com o **MLflow Tracking**, possibilitando registrar parâmetros, métricas e artefatos de cada execução, acompanhando a evolução dos experimentos ao longo do projeto.