# Model Evaluation & Selection

A etapa de Model Evaluation & Selection é responsável por comparar os modelos treinados e registrados durante o pipeline, utilizando métricas de desempenho para fundamentar a escolha do candidato mais adequado aos objetivos do projeto.

Em um fluxo de Machine Learning, o treinamento e a otimização de hiperparâmetros não garantem, por si só, que determinado modelo seja a melhor opção. Diferentes algoritmos e configurações podem apresentar resultados distintos, tornando necessária uma avaliação sistemática dos candidatos. No contexto de MLOps, essa etapa contribui para a rastreabilidade, a transparência e a organização do processo de tomada de decisão, permitindo que a seleção seja fundamentada em métricas e critérios previamente definidos.

Neste projeto, os modelos registrados no MLflow Model Registry foram comparados com base nos resultados obtidos durante o treinamento, a avaliação e a otimização. O objetivo desta etapa é avaliar os modelos disponíveis no MLflow Model Registry e identificar o candidato prioritário para as próximas etapas do pipeline.

A avaliação considera métricas obtidas no conjunto de teste, além dos resultados de validação cruzada disponíveis para os modelos otimizados.

## Modelos avaliados

Foram considerados quatro modelos previamente registrados no MLflow Model Registry, provenientes das etapas de baseline e otimização.

| Modelo | Origem |
|---|---|
| Random Forest Baseline | Treinamento inicial |
| Logistic Regression Baseline | Treinamento inicial |
| RandomizedSearchCV | Otimização de hiperparâmetros |
| HalvingGridSearchCV | Otimização de hiperparâmetros |

Cada modelo possui uma versão registrada no MLflow Model Registry, vinculada ao Run de origem, permitindo recuperar suas métricas e rastrear os resultados obtidos durante as etapas anteriores.

A seleção dos modelos foi realizada utilizando critérios hierárquicos, definidos de acordo com a prioridade das métricas para o contexto do projeto.

### 4.1 Recall da classe Malignant

O Recall da classe Malignant foi definido como o principal critério de seleção.

Essa métrica representa a proporção de casos malignos corretamente identificados pelo modelo em relação ao total de casos malignos existentes no conjunto de avaliação.

No contexto do dataset Breast Cancer Wisconsin, a prioridade atribuída ao Recall Malignant busca reduzir a quantidade de casos malignos classificados incorretamente como benignos.

Entretanto, essa prioridade representa um critério experimental de seleção e não uma validação de desempenho clínico.

### 4.2 F1 Macro

O F1 Macro corresponde à média aritmética do F1-score calculado individualmente para cada classe, atribuindo o mesmo peso a ambas.

Essa métrica permite avaliar o equilíbrio do desempenho do modelo entre as classes, evitando que o desempenho de uma classe domine a avaliação agregada.

### 4.3 Accuracy

A Accuracy representa a proporção de classificações corretas em relação ao total de observações avaliadas.

Foi utilizada como terceiro critério de seleção, considerando o desempenho geral do modelo no conjunto de teste.

### 4.4 Hierarquia de seleção

Os modelos foram ordenados utilizando os seguintes critérios, em ordem de prioridade:

1. Maior Recall Malignant.
2. Maior F1 Macro.
3. Maior Accuracy.

A ordenação é hierárquica, ou seja, os critérios subsequentes são utilizados para desempate quando os valores do critério prioritário são iguais.

Não foi utilizada uma ponderação matemática entre as métricas.

### 4.5 Cross-validation

A métrica CV F1 Macro foi incluída na comparação para contextualizar os resultados de validação cruzada dos modelos otimizados.

Essa métrica permite observar o desempenho médio obtido durante os processos de validação utilizados na otimização de hiperparâmetros.

Como a métrica não está disponível nos Runs dos modelos baseline, ela não foi utilizada como critério de ordenação, evitando penalizar os modelos pela ausência desse registro.

## 5. Integração com MLflow

A avaliação utiliza as informações disponibilizadas pelo MLflow Tracking e pelo MLflow Model Registry.

### MLflow Tracking

O MLflow Tracking é responsável pelo armazenamento e rastreamento dos experimentos, incluindo parâmetros, métricas, artefatos e informações associadas aos Runs.

Durante esta etapa, os Runs são consultados para recuperar as métricas de desempenho previamente registradas, evitando a necessidade de repetir os treinamentos.

### MLflow Model Registry

O MLflow Model Registry centraliza os modelos registrados e suas respectivas versões.

A partir do Registry, são recuperadas as versões disponíveis do modelo `breast-cancer-classifier`, permitindo identificar os modelos candidatos e relacioná-los aos Runs de origem.

Essa integração permite que a avaliação seja realizada sobre os modelos já registrados, mantendo a rastreabilidade entre experimentos, métricas e versões.

## 6. Implementação

A etapa foi implementada no módulo:

`src/mlops/evaluation/model_selection.py`

O módulo concentra as funções responsáveis pela recuperação das versões, consulta das métricas, construção da comparação, aplicação dos critérios de seleção e geração do relatório.

### 6.1 load_registered_versions()

Responsável por consultar o MLflow Model Registry e recuperar as versões registradas do modelo `breast-cancer-classifier`.

A função utiliza o `MlflowClient` para consultar as versões disponíveis, permitindo que os candidatos sejam recuperados diretamente do Registry.

### 6.2 get_model_metrics()

Responsável por recuperar as métricas armazenadas no MLflow Tracking a partir do Run ID associado a cada versão registrada.

Essa função permite acessar os resultados das avaliações anteriores sem repetir os treinamentos.

### 6.3 build_comparison_table()

Responsável por construir uma tabela comparativa utilizando Pandas.

Para cada versão registrada, são recuperadas as informações relevantes para a avaliação, incluindo:

- Número da versão no Registry.
- Nome do modelo.
- Run ID.
- Test Accuracy.
- Test F1 Macro.
- Test Recall Malignant.
- CV F1 Macro, quando disponível.

A tabela consolidada fornece uma estrutura padronizada para comparação entre os candidatos.

### 6.4 evaluate_candidates()

Responsável por aplicar os critérios de avaliação e ordenação dos modelos.

A função organiza os candidatos de acordo com a hierarquia definida:

1. Test Recall Malignant.
2. Test F1 Macro.
3. Test Accuracy.

Após a ordenação, os candidatos recebem uma classificação de acordo com sua posição na comparação, permitindo identificar o candidato prioritário.

### 6.5 select_best_model()

Responsável por selecionar o primeiro candidato da tabela ordenada, conforme os critérios definidos na função de avaliação.

A função retorna o candidato prioritário identificado no processo de comparação, permitindo que suas informações sejam utilizadas na geração do relatório.

### 6.6 generate_evaluation_report()

Responsável por gerar o relatório de avaliação em Markdown, consolidando os resultados da comparação e a decisão fundamentada nos critérios definidos.

O relatório apresenta:

- Objetivo da avaliação.
- Critérios de seleção.
- Resultados comparativos.
- Classificação dos candidatos.
- Modelo selecionado.
- Métricas do candidato prioritário.
- Justificativa técnica.
- Observações e limitações.
- Próximas etapas do pipeline.

O relatório é armazenado em:

`reports/model_evaluation/model_selection_report.md`

### 6.7 Fluxo de execução

O fluxo de execução da etapa é organizado da seguinte forma:

1. Recuperação das versões registradas no MLflow Model Registry.
2. Consulta dos Runs associados às versões.
3. Recuperação das métricas armazenadas no MLflow Tracking.
4. Construção da tabela comparativa.
5. Aplicação dos critérios hierárquicos de avaliação.
6. Seleção do candidato prioritário.
7. Geração do relatório de avaliação.

## 7. Resultados da avaliação

Os resultados detalhados da comparação, incluindo as métricas, a classificação dos candidatos e o modelo selecionado, estão apresentados no relatório gerado pela execução do módulo.

O relatório consolidado desta etapa é:

`reports/model_evaluation/model_selection_report.md`

A seleção do candidato prioritário é fundamentada nos critérios definidos previamente, permitindo rastrear a decisão até os resultados armazenados no MLflow Tracking e as versões disponíveis no Model Registry.

## 8. Model Registry e identificação do modelo selecionado

Após a avaliação, o candidato prioritário pode ser identificado no MLflow Model Registry por meio da atribuição de um alias, como `champion`.

O alias permite identificar uma versão específica do modelo sem depender diretamente do número da versão, facilitando seu consumo em etapas posteriores do pipeline.

A atribuição do alias representa a identificação operacional de uma versão selecionada. Ela não substitui a avaliação, nem representa, por si só, uma validação clínica ou aprovação para utilização em sistemas de saúde.

A atualização do alias pode ser realizada diretamente na interface do MLflow Model Registry, permitindo alterar a versão designada conforme novas avaliações sejam realizadas.

## 9. Limitações

A seleção realizada nesta etapa corresponde a uma comparação experimental entre os modelos avaliados, utilizando os resultados registrados no pipeline.

Embora a priorização do Recall Malignant tenha sido definida como critério principal, a escolha de um modelo para aplicações relacionadas à saúde exige análises adicionais, incluindo validação independente, avaliação de erros, análise de riscos e validação clínica.

Os resultados obtidos neste projeto são utilizados para fins de estudo, desenvolvimento e demonstração de práticas de MLOps, não representando uma solução validada para diagnóstico clínico.

## 10. Próxima etapa

Após a avaliação e seleção dos modelos, o pipeline avança para a etapa de Data Drift Monitoring.

O monitoramento busca identificar possíveis alterações na distribuição dos dados utilizados pelo modelo, contribuindo para o acompanhamento de mudanças nos dados ao longo do tempo e para a futura implementação de estratégias de monitoramento de modelos em produção.
