![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data%20Visualization-11557C?logo=plotly&logoColor=white)

# Redes Neurais Artificiais: Previsão de Valores de Seguros
## Funções de ativação e Regressão

Este estudo tem como objetivo analisar como diferentes funções de ativação influenciam o comportamento de uma rede neural artificial durante o aprendizado de máquina. O estudo utiliza Python e PyTorch, com foco na regressão e na influência das funções de ativação sobre o comportamento do modelo.

O experimento utiliza um dataset sintético de 100 segurados fictícios, contendo nome, idade e valor do seguro automotivo. Parte dos valores dos seguros estará ausente, simulando uma situação em que uma empresa precisa recuperar informações de sua base de dados.

A proposta é treinar uma rede neural para aprender a relação entre a idade dos segurados e os valores conhecidos dos seguros, realizar previsões para os registros incompletos e avaliar a qualidade dos resultados por meio de métricas de validação.

O foco principal será comparar o comportamento das funções de ativação e observar como elas influenciam a curva aprendida pelo modelo, a capacidade de representar a relação entre as variáveis e a qualidade das previsões.

O experimento vai investigar se a rede consegue aproximar o comportamento observado nos dados. A forma real da curva dependerá do dataset, da arquitetura e do treinamento.

## Cenário do negócio

A **Vértice Seguros** é uma empresa fictícia do setor de seguros automotivos que está revisando sua base cadastral após uma migração de sistema. Durante o processo, alguns valores de apólices ficaram ausentes, embora os nomes e as idades dos segurados tenham sido preservados.

Para investigar uma possível solução, a equipe de dados decidiu desenvolver um experimento com redes neurais artificiais, utilizando a idade como variável de entrada e o valor do seguro como variável-alvo.

O objetivo é verificar se o modelo consegue aprender padrões presentes nos dados disponíveis e estimar os valores ausentes com um nível de erro aceitável.

> **Observação:** este é um estudo educacional com dados sintéticos. As previsões não representam cotações reais e não devem ser utilizadas para definir preços comerciais de seguros.

## Objetivos do estudo

* Compreender a estrutura de uma rede neural artificial aplicada à regressão.
* Investigar como pesos (*weights*) e bias influenciam as transformações realizadas pela rede.
* Estudar a influência de duas funções de ativação no ajuste do modelo.
* Aplicar a normalização da variável de entrada.
* Treinar o modelo utilizando os registros com valores de seguro conhecidos.
* Prever os valores ausentes do dataset.
* Avaliar o desempenho do modelo por meio de métricas de regressão.
* Comparar as previsões da rede neural com uma abordagem de referência.
* Identificar limitações e oportunidades de melhoria do experimento.

## Dataset

O dataset será composto por 100 observações e três colunas:

| Coluna         | Descrição                           | Tipo                           |
| -------------- | ----------------------------------- | ------------------------------ |
| `nome`         | Nome do segurado                    | Texto                          |
| `idade`        | Idade do segurado em anos           | Numérico                       |
| `valor_seguro` | Valor do seguro automotivo em reais | Numérico, com valores ausentes |

Os dados serão armazenados inicialmente em sua escala original, sem normalização. Alguns registros da coluna `valor_seguro` conterão valores ausentes (`NaN`), que serão utilizados como casos para previsão.

Exemplo ilustrativo:

| nome         | idade | valor_seguro |
| ------------ | ----: | -----------: |
| Ana Martins  |    19 |      3850.00 |
| Bruno Costa  |    28 |      2400.00 |
| Carla Souza  |    42 |      1650.00 |
| Daniel Lima  |    61 |      2300.00 |
| Elisa Rocha  |    35 |        `NaN` |
| Felipe Alves |    24 |        `NaN` |

## Metodologia

O desenvolvimento será organizado nas seguintes etapas:

### 1. Geração dos dados

Criação de um dataset sintético com 100 segurados fictícios, contendo nomes, idades e valores de seguro. Parte dos valores será definida como ausente para simular o problema de recuperação de dados.

O arquivo original será preservado para manter os dados em sua escala natural e permitir a reprodução do experimento.

### 2. Preparação dos dados

A coluna `idade` será utilizada como variável de entrada (*input*), enquanto `valor_seguro` será a variável-alvo (*target*).

A coluna `nome` será mantida para identificação dos registros, mas não será utilizada como entrada numérica da rede neural.

A preparação incluirá:

* Separação dos registros com valores conhecidos e ausentes.
* Divisão dos registros conhecidos em conjuntos de treinamento, validação e teste.
* Normalização da idade, com parâmetros calculados exclusivamente a partir do conjunto de treinamento.
* Tratamento adequado da escala do valor-alvo, caso seja necessário para a arquitetura escolhida.

### 3. Construção da rede neural

A rede neural será implementada utilizando PyTorch, com uma variável de entrada, transformações intermediárias e uma saída numérica correspondente à estimativa do valor do seguro.

O estudo investigará a influência de duas funções de ativação sobre a curva aprendida pelo modelo e sua capacidade de representar a relação entre idade e valor do seguro.

Os pesos e o bias serão ajustados durante o treinamento para reduzir a diferença entre os valores previstos e os valores reais conhecidos.

### 4. Treinamento do modelo

O modelo será treinado utilizando os registros que possuem valores de seguro conhecidos.

Durante o treinamento, serão definidos os seguintes componentes:

* Número de épocas (*epochs*).
* Taxa de aprendizado (*learning rate*).
* Função de perda (*loss function*).
* Otimizador responsável pela atualização dos parâmetros.
* Acompanhamento da perda durante o treinamento e a validação.

A função de perda será escolhida de acordo com a tarefa de regressão, permitindo medir o erro entre as estimativas e os valores reais.

### 5. Avaliação e validação

O desempenho será avaliado em registros conhecidos que não tenham sido utilizados para ajustar os parâmetros do modelo.

Serão utilizadas métricas de regressão, como:

| Métrica                                   | Objetivo                                                                       |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| MAE (*Mean Absolute Error*)               | Medir o erro absoluto médio das previsões, em reais                            |
| RMSE (*Root Mean Squared Error*)          | Penalizar erros maiores e expressar o resultado na escala do seguro            |
| Gráfico de valores reais versus previstos | Avaliar visualmente a proximidade entre as estimativas e os valores observados |

Também será realizada uma comparação com um modelo de referência simples, como a previsão pela média dos valores de seguro do conjunto de treinamento.

Essa comparação permitirá verificar se a rede neural oferece algum ganho real em relação a uma abordagem mais simples.

### 6. Previsão dos valores ausentes

Após a avaliação, o modelo selecionado será utilizado para estimar os valores ausentes do dataset.

As previsões serão associadas aos respectivos segurados e armazenadas em um arquivo separado, preservando o dataset original.

Os valores estimados serão tratados como previsões do modelo, e não como valores reais confirmados.

## Fluxo do experimento:

![Fluxo](analytics/graficos/fluxo_estudo_ml.png)

## Tecnologias utilizadas

* **Python:** desenvolvimento do experimento.
* **PyTorch:** construção, treinamento e avaliação da rede neural.
* **Pandas:** criação, manipulação e organização do dataset.
* **NumPy:** operações numéricas e geração dos dados sintéticos.
* **Matplotlib:** visualização dos dados, das curvas aprendidas e dos resultados.
* **Scikit-learn:** apoio à divisão dos dados e ao cálculo de métricas de regressão.
* **MkDocs:** documentação do estudo e publicação no GitHub Pages.

## Limitações do estudo

O valor real de um seguro automotivo depende de diversos fatores, como modelo do veículo, localização, cobertura contratada e histórico de sinistros. Portanto, a idade, isoladamente, pode não explicar adequadamente o preço.

Além disso, o dataset será sintético e representará um cenário controlado para fins educacionais. Um bom resultado nesse conjunto não garante desempenho equivalente em dados reais.

A qualidade das previsões será analisada considerando as métricas de validação, a comparação com o modelo de referência e a capacidade de generalização para registros não utilizados no treinamento.

## Conclusão

O projeto demonstrou a aplicação de uma rede neural desenvolvida com PyTorch para estimar valores de seguros a partir da idade dos clientes, incluindo a previsão de 15 valores ausentes no conjunto de dados.

A avaliação dos 85 registros com valores conhecidos apresentou MAE de R$ 226,31, RMSE de R$ 275,43 e R² de 0,883239. Esses resultados permitiram analisar o ajuste do modelo e compreender a importância das métricas na avaliação de modelos preditivos.

Além da implementação técnica, o estudo proporcionou a aplicação prática de conceitos de preparação de dados, normalização, tensores, treinamento de redes neurais e análise de resultados, consolidando conhecimentos em **Python, PyTorch e Deep Learning**.

## Autor:

**Daniel Martins França**


