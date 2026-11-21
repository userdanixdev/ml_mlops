# Otimização de Hiperparâmetros

A otimização de hiperparâmetros é uma etapa do desenvolvimento de modelos de Machine Learning que busca encontrar as melhores configurações possíveis para um algoritmo, com o objetivo de melhorar seu desempenho preditivo.

Diferentemente dos parâmetros internos, que são aprendidos durante o processo de treinamento, os hiperparâmetros são definidos pelo desenvolvedor ou selecionados por métodos de busca.

No caso do algoritmo **Random Forest**, alguns exemplos de hiperparâmetros são:

- `n_estimators`: quantidade de árvores utilizadas na floresta.
- `max_depth`: profundidade máxima permitida para cada árvore.
- `min_samples_split`: quantidade mínima de amostras necessárias para dividir um nó.
- `min_samples_leaf`: quantidade mínima de amostras exigida em um nó folha.
- `max_features`: quantidade de características consideradas em cada divisão.

*Cada configuração pode produzir resultados diferentes, influenciando a capacidade de generalização do modelo.*

> Dessa forma, a otimização de hiperparâmetros permite explorar diferentes configurações de um algoritmo para identificar aquelas que apresentam melhor desempenho.

Entre seus principais objetivos estão:

- Melhorar as métricas de avaliação do modelo.
- Reduzir o risco de overfitting e underfitting.
- Encontrar configurações que generalizem melhor para dados não utilizados no treinamento.
- Comparar diferentes combinações de hiperparâmetros de maneira sistemática.
- Tornar o processo de treinamento mais consistente e reproduzível.

**É importante destacar que uma configuração com desempenho superior nos dados de treinamento não significa necessariamente um modelo melhor. Por isso, a avaliação deve considerar validação cruzada e dados de teste separados.**

## Como funciona o processo?

De maneira geral, o processo de otimização segue estas etapas:

1. **Definição do algoritmo:** escolha do modelo que será otimizado.
2. **Definição do espaço de busca:** seleção dos hiperparâmetros e dos valores que serão testados.
3. **Execução da busca:** avaliação das diferentes combinações de configurações.
4. **Validação cruzada:** divisão dos dados de treinamento em diferentes partes para avaliar a estabilidade do modelo.
5. **Seleção da melhor configuração:** identificação da combinação que apresenta o melhor resultado segundo a métrica escolhida.
6. **Avaliação final:** aplicação do modelo otimizado aos dados de teste, que não participaram da busca.

Neste projeto, serão utilizados dois métodos de otimização disponibilizados pelo Scikit-learn:

### RandomizedSearchCV

O `RandomizedSearchCV` seleciona aleatoriamente combinações de hiperparâmetros dentro de um espaço de busca definido.

Em vez de testar todas as combinações possíveis, o método avalia uma quantidade determinada de configurações, utilizando validação cruzada para comparar os resultados.

> Essa abordagem permite explorar espaços de busca maiores com um custo computacional potencialmente menor.

### HalvingGridSearchCV

O `HalvingGridSearchCV` utiliza uma estratégia de eliminação sucessiva (*successive halving*).

Inicialmente, avalia diversas combinações de hiperparâmetros utilizando uma quantidade menor de recursos. Nas etapas seguintes, elimina as configurações com pior desempenho e direciona mais recursos para as combinações mais promissoras.

> Essa estratégia busca reduzir o custo computacional da otimização, concentrando os recursos nas configurações com maior potencial.

Neste projeto, será utilizada a métrica **F1-score macro** como critério de seleção durante a otimização.

O F1-score combina precisão (*precision*) e revocação (*recall*) em uma única medida.

A média macro calcula o F1-score de cada classe individualmente e depois realiza a média simples entre elas, atribuindo o mesmo peso às classes.

Essa escolha permite avaliar o desempenho do modelo considerando igualmente as classes `malignant` e `benign`.

Assim, a otimização de hiperparâmetros contribui para a construção de modelos mais consistentes e para a melhoria do processo de desenvolvimento.

No contexto deste projeto, os resultados obtidos serão utilizados nas próximas etapas do pipeline:

- Comparação entre os modelos baseline e otimizados.
- Registro de parâmetros, métricas e resultados com MLflow Tracking.
- Seleção do modelo com melhor desempenho.
- Versionamento e gerenciamento do modelo por meio do MLflow Model Registry.

*Dessa forma, a otimização não representa apenas uma tentativa de melhorar uma métrica, mas uma etapa estruturada do ciclo de vida do modelo de Machine Learning.*