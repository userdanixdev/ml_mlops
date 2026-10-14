# Pipeline de classificação

## Visão geral

O pipeline foi desenvolvido para demonstrar um fluxo completo de classificação binária, desde a geração de dados sintéticos até a exportação e validação das previsões.

## Fluxo de execução

![Fluxo completo](/docs/images/fluxo_classificacao_academica.png)

## 1. Geração dos dados

O NumPy é utilizado para gerar dados sintéticos relacionados a três características dos estudantes:

- `estudou`
- `conhecimento_previo`
- `facilidade_aprendizado`

A base é organizada em um DataFrame com Pandas e exportada para `alunos_base_original.xlsx`.

Parte dos registros é gerada sem resultado conhecido, simulando um cenário de dados incompletos.

## 2. Preparação dos dados

A base original é carregada e dividida em dois grupos:

- Registros com resultado conhecido, utilizados no treinamento e na avaliação.
- Registros sem resultado, destinados à geração de previsões.

As características de entrada e os resultados conhecidos são convertidos em tensores para utilização no PyTorch.

## 3. Treinamento do modelo

O modelo utiliza uma camada linear com três entradas e uma saída, produzindo um valor de classificação.

A função `BCEWithLogitsLoss` calcula a perda durante o treinamento, enquanto o otimizador Adam atualiza os parâmetros do modelo.

A saída do modelo é convertida em probabilidade por meio da função sigmoid. Um limiar de classificação permite transformar a probabilidade em uma classe binária.

## 4. Geração das previsões

Após o treinamento, o modelo gera previsões para os registros que não possuem resultado conhecido.

As previsões e probabilidades são associadas aos respectivos registros, preservando os dados originais.

## 5. Exportação dos resultados

O Pandas organiza os resultados em um novo DataFrame e exporta o arquivo `alunos_com_predicoes.xlsx`.

O arquivo inclui:

- `resultado`: resultado original, incluindo valores ausentes.
- `resultado_predito`: resultado conhecido ou previsto.
- `probabilidade_predicao`: probabilidade associada à previsão dos registros originalmente sem resultado.

## 6. Validação do arquivo

Uma etapa específica verifica:

- Quantidade de registros.
- Presença de valores ausentes.
- Validade das classes previstas.
- Preenchimento das previsões para os registros sem resultado.
- Probabilidades dentro do intervalo de 0 a 1.

## 7. Avaliação independente

Um script separado divide os registros conhecidos em conjuntos de treinamento e teste, permitindo avaliar o desempenho do modelo em dados que não foram utilizados para ajustar seus parâmetros.

Os resultados são apresentados na página [Avaliação do modelo](avaliacao.md).

## Observação

A avaliação independente é realizada em um treinamento separado. Ela mede o desempenho do modelo no conjunto de teste e não deve ser confundida com as previsões produzidas para os registros sem resultado.