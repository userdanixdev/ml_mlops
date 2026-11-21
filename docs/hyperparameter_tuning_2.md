# HalvingGridSearchCV

O `HalvingGridSearchCV` é uma técnica de otimização de hiperparâmetros disponibilizada pelo Scikit-learn, baseada no algoritmo **Successive Halving (Eliminação Sucessiva)**.

Seu objetivo é encontrar uma combinação de hiperparâmetros que proporcione o melhor desempenho ao modelo, utilizando os recursos computacionais de maneira progressiva.

Diferentemente do `RandomizedSearchCV`, que seleciona aleatoriamente uma quantidade determinada de combinações de hiperparâmetros, o `HalvingGridSearchCV` avalia inicialmente todas as combinações definidas no espaço de busca e elimina progressivamente aquelas que apresentam resultados inferiores.

**Os candidatos com melhor desempenho avançam para as próximas iterações, recebendo mais recursos computacionais até a seleção da melhor configuração encontrada.**

O processo de otimização acontece em etapas:

1. **Definição dos candidatos:** são geradas as combinações possíveis de hiperparâmetros a partir do espaço de busca definido.
2. **Avaliação inicial:** todas as combinações são avaliadas utilizando uma quantidade inicial reduzida de recursos.
3. **Eliminação:** os candidatos com desempenho inferior são descartados.
4. **Aumento de recursos:** os candidatos selecionados recebem mais recursos na próxima iteração.
5. **Seleção final:** o processo continua até identificar a melhor combinação de hiperparâmetros.

Considerando uma busca com 27 combinações e `factor=3`, o processo poderia ocorrer da seguinte maneira:

| Iteração | Combinações avaliadas | Recursos por candidato |
|---|---:|---|
| 1 | 27 | 100 |
| 2 | 9 | 300 |
| 3 | 3 | 900 |

*Valores meramente ilustrativos. A quantidade real de candidatos, recursos e iterações depende das configurações da busca.*

> A cada iteração, aproximadamente dois terços dos candidatos são eliminados, enquanto os restantes recebem mais recursos para uma avaliação aprofundada.


| Parâmetro | Descrição |
|---|---|
| `estimator` | Modelo que será otimizado, neste projeto o Random Forest. |
| `param_grid` | Dicionário contendo os hiperparâmetros e seus respectivos valores para avaliação. |
| `factor` | Define a proporção de candidatos eliminados a cada iteração. |
| `resource` | Define o recurso utilizado e aumentado durante as iterações. |
| `min_resources` | Quantidade mínima de recursos utilizada na primeira iteração. |
| `max_resources` | Quantidade máxima de recursos permitida durante a busca. |
| `cv` | Define a estratégia de validação cruzada. |
| `scoring` | Métrica utilizada para avaliar o desempenho dos modelos. |
| `n_jobs` | Define a quantidade de processos paralelos utilizados. |
| `refit` | Quando habilitado, treina novamente o melhor modelo utilizando todo o conjunto de treinamento. |

## Resultados da otimização

Após a execução da busca, o objeto `HalvingGridSearchCV` disponibiliza atributos importantes para análise:

- `best_params_`: retorna a melhor combinação de hiperparâmetros encontrada.
- `best_score_`: retorna a melhor pontuação média obtida na validação cruzada durante a última iteração.
- `best_estimator_`: retorna o modelo configurado com os melhores hiperparâmetros.
- `cv_results_`: disponibiliza os resultados detalhados das avaliações realizadas durante a busca.
- `n_iterations_`: informa a quantidade de iterações executadas.
- `n_candidates_`: informa a quantidade de candidatos avaliados em cada iteração.
- `n_resources_`: informa a quantidade de recursos utilizados em cada iteração.

**Esses atributos permitem analisar o comportamento da otimização, identificar os melhores hiperparâmetros e comparar os resultados com outras estratégias de busca.**

## Aplicação no projeto

Neste projeto, o `HalvingGridSearchCV` será utilizado como uma segunda estratégia de otimização do modelo **Random Forest**, complementando a busca realizada anteriormente com `RandomizedSearchCV`.

Ambas as técnicas utilizarão o dataset `Breast Cancer Wisconsin` e os dados de treinamento previamente preparados. O objetivo é avaliar qual estratégia consegue encontrar a configuração de hiperparâmetros com melhor desempenho para o problema de classificação.

> Os resultados serão armazenados em arquivos CSV, permitindo a geração de relatórios e gráficos comparativos.

*Além disso, os experimentos serão registrados no **MLflow Tracking**, possibilitando acompanhar os parâmetros utilizados, as métricas obtidas e os resultados de cada estratégia.*

Após a comparação, o melhor modelo poderá seguir para as próximas etapas do pipeline:

- Avaliação final do modelo.
- Registro e versionamento no MLflow Model Registry.
- Simulação de Data Drift.
- Disponibilização do modelo por meio de uma API com FastAPI.
- Containerização com Docker.
- Deploy no Kubernetes.

## Considerações

O `HalvingGridSearchCV` permite otimizar o processo de busca por meio da eliminação progressiva de candidatos com desempenho inferior, reduzindo o investimento computacional em combinações menos promissoras.

Entretanto, como a eliminação acontece nas primeiras iterações, uma combinação potencialmente boa pode ser descartada prematuramente. Por isso, é importante configurar adequadamente os recursos iniciais e o fator de eliminação.

**Observação:** no Scikit-learn, o `HalvingGridSearchCV` ainda é considerado um recurso experimental. Sua utilização exige habilitação explícita por meio do módulo `sklearn.experimental`.

## Referências

- [Scikit-learn – HalvingGridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html)
- [Scikit-learn – Successive Halving](https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving)