# RandomizedSearchCV

O `RandomizedSearchCV`, disponibilizado pelo **Scikit-learn**, é uma técnica de otimização de hiperparâmetros que realiza uma busca aleatória por combinações de configurações dentro de um espaço de busca previamente definido.

Diferentemente de uma busca exaustiva, que avalia todas as combinações possíveis, o `RandomizedSearchCV` seleciona uma quantidade determinada de combinações, permitindo explorar diferentes configurações com menor custo computacional.

Neste projeto, a técnica foi aplicada ao algoritmo `Random Forest`, buscando identificar uma configuração com melhor capacidade de classificação para o dataset `Breast Cancer Wisconsin.`

## Configurações utilizadas

A implementação utiliza as seguintes configurações:

- **Random Forest:** algoritmo selecionado para a otimização.
- **Espaço de busca:** conjunto de valores possíveis para os hiperparâmetros.
- **n_iter = 20:** quantidade de combinações avaliadas aleatoriamente.
- **cv = 5:** validação cruzada com cinco divisões (*folds*).
- **scoring = f1_macro:** métrica utilizada para comparar as configurações.
- **random_state = 42:** garante a reprodutibilidade da seleção aleatória.
- **n_jobs = -1:** permite utilizar os processadores disponíveis para paralelizar a busca.
- **verbose = 2:** apresenta informações sobre o andamento da otimização no terminal.

### Validação cruzada

A validação cruzada com cinco divisões permite avaliar cada configuração em diferentes partições dos dados de treinamento.

Em cada rodada, uma parte dos dados é utilizada para validação e as demais para treinamento. Ao final, os resultados são agregados para produzir uma avaliação mais consistente.

### Métrica F1-score macro

O projeto utiliza o F1-score macro como critério de seleção.

Essa métrica calcula o F1-score individualmente para cada classe e realiza uma média simples entre elas, atribuindo o mesmo peso às classes `malignant` e `benign`.

## Avaliação do melhor modelo

Após a execução da busca, o atributo `best_estimator_` retorna o modelo configurado com os melhores hiperparâmetros encontrados.

Esse modelo é avaliado utilizando o conjunto de teste, que permanece separado dos dados utilizados durante a otimização.

Para essa avaliação, são calculadas métricas como:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- Matriz de confusão.

> A separação entre treinamento, validação e teste ajuda a evitar que o desempenho final seja avaliado sobre os mesmos dados utilizados na seleção dos hiperparâmetros.

[avaliaçao_melhor_modelo]()

## Persistência do modelo com Joblib

Após a otimização e avaliação, o melhor modelo é persistido utilizando a biblioteca Joblib.

O Joblib permite serializar objetos Python, incluindo modelos treinados do Scikit-learn, possibilitando armazená-los em disco e recuperá-los posteriormente sem a necessidade de executar novamente todo o treinamento.

O modelo otimizado é salvo com identificação própria:

`models/model_3_random_forest_optimized.joblib`

Essa organização mantém os modelos baseline separados do modelo otimizado e facilita sua utilização nas próximas etapas do pipeline.

**A persistência também contribui para a reprodutibilidade, permitindo reutilizar o modelo treinado em processos de inferência e implantação.**

## Armazenamento das métricas em CSV

As métricas de avaliação são armazenadas no arquivo:

`reports/model_metrics.csv`

O arquivo utiliza formato CSV para organizar os resultados de diferentes modelos em uma estrutura tabular. A implementação utiliza uma função compartilhada para salvar as métricas, identificando cada modelo pelo nome. Caso já exista um registro com a mesma identificação, ele é atualizado, preservando os resultados dos demais modelos.

Essa abordagem permite manter em um único arquivo os resultados dos modelos baseline e otimizados.

> O CSV também serve como fonte de dados para a geração de gráficos comparativos, permitindo visualizar a evolução das métricas ao longo do desenvolvimento.

## Geração dos resultados da otimização em CSV

Além das métricas finais, os resultados completos da busca são armazenados em:

`reports/optimization/randomized_search_results.csv`

Esse arquivo é gerado a partir do atributo `cv_results_` do `RandomizedSearchCV.`

Ele contém informações sobre as combinações de hiperparâmetros avaliadas, suas pontuações de validação e os tempos de execução. Com esses dados, é possível realizar análises adicionais e construir visualizações para comparar as configurações testadas.

A separação entre o CSV de métricas finais e o CSV de resultados da busca permite atender a duas necessidades diferentes:

- Comparar o desempenho dos modelos treinados.
- Investigar o comportamento das combinações de hiperparâmetros durante a otimização.

## Geração de relatório em Markdown

Para documentar os resultados da execução, a implementação gera automaticamente um relatório em Markdown:

`reports/optimization/randomized_search_report.md`

O relatório reúne informações relevantes do processo, como:

- Melhores hiperparâmetros encontrados;
- Melhor pontuação obtida na validação cruzada;
- Accuracy no conjunto de teste;
- Relatório de classificação;
- Matriz de confusão.

*O relatório pode ser consultado diretamente no projeto, sem depender exclusivamente da saída apresentada no terminal.*

## Saída no terminal

Durante a execução, o parâmetro `verbose = 2` apresenta o andamento das avaliações realizadas pelo RandomizedSearchCV.

Ao final, também são exibidas informações sobre os melhores hiperparâmetros e as métricas de avaliação.

Dessa forma, o projeto mantém três formas complementares de disponibilizar os resultados:

1. **Terminal:** acompanhamento imediato da execução.
2. **Markdown:** documentação legível dos resultados.
3. **CSV:** armazenamento estruturado para análises e visualizações.

## Os artefatos

A implementação mantém os arquivos gerados em diretórios específicos:

| Arquivo | Finalidade |
|---|---|
| `models/model_3_random_forest_optimized.joblib` | Persistência do modelo otimizado |
| `reports/model_metrics.csv` | Métricas comparativas dos modelos |
| `reports/optimization/randomized_search_results.csv` | Resultados das combinações avaliadas |
| `reports/optimization/randomized_search_report.md` | Relatório detalhado da execução |

*Essa organização separa os artefatos do treinamento, os resultados da otimização e a documentação, facilitando a manutenção e a evolução do pipeline.*

# Relatório de Otimização - RandomizedSearchCV

## Configuração da busca

- Modelo: Random Forest
- Método: RandomizedSearchCV
- Número de combinações: 20
- Validação cruzada: 5 folds
- Métrica de seleção: F1-macro
- Random state: 42

## Melhores hiperparâmetros

```python
{'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 10}
```

## Resultados

- Melhor F1-macro na validação cruzada: 0.9554
- Accuracy no conjunto de teste: 0.9561

## Matriz de confusão

```text
[[39  3]
 [ 2 70]]
```

## Métricas de classificação

```text
              precision    recall  f1-score    support
malignant      0.951220  0.928571  0.939759   42.00000
benign         0.958904  0.972222  0.965517   72.00000
accuracy       0.956140  0.956140  0.956140    0.95614
macro avg      0.955062  0.950397  0.952638  114.00000
weighted avg   0.956073  0.956140  0.956027  114.00000
```
## Integração com o fluxo MLOps

O `RandomizedSearchCV` representa uma etapa importante na evolução dos modelos desenvolvidos neste projeto.

Os artefatos gerados serão utilizados nas etapas seguintes:

- Comparação dos modelos baseline e otimizados.
- Registro de parâmetros, métricas e artefatos no MLflow Tracking.
- Seleção do modelo candidato para registro.
- Versionamento do modelo no MLflow Model Registry.
- Preparação do modelo para disponibilização por meio de uma API.

> Assim, a otimização passa a fazer parte de um fluxo estruturado, no qual treinamento, avaliação, persistência e documentação são tratados de maneira organizada e reproduzível.