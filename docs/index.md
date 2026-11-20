# Estudos Sobre Machine Learning

Este portfólio reúne uma série de estudos práticos sobre **Machine Learning e Deep Learning**, desenvolvidos com Python, com o objetivo de consolidar conhecimentos teóricos e explorar, na prática, as diferentes etapas envolvidas na construção de modelos de aprendizado de máquina.

A proposta é combinar fundamentos conceituais com experimentação e implementação, abordando desde a preparação e análise dos dados até o treinamento, avaliação e utilização de modelos preditivos.

Os estudos estão organizados em diferentes etapas de evolução, utilizando conjuntos de dados sintéticos e reais, algoritmos tradicionais de Machine Learning e redes neurais artificiais.

O portfólio foi desenvolvido progressivamente, explorando diferentes problemas e abordagens de aprendizado de máquina.

### 1. Classificação binária de desempenho acadêmico

Estudo introdutório de classificação binária utilizando `PyTorch`, com geração de dados sintéticos para prever resultados acadêmicos.

Foram explorados conceitos de preparação dos dados, construção e treinamento de modelos, inferência e avaliação por métricas de classificação.

**Principais conceitos:** classificação binária, tensores, função de perda, otimização, inferência e matriz de confusão.

[Consultar estudo ML - Desempenho Acadêmico](https://userdanixdev.github.io/ml_mlops/){ .md-button }

### 2. Classificação de padrões não lineares - Dataset Moon

Desenvolvimento de uma rede neural artificial utilizando TensorFlow e Keras para resolver um problema de classificação binária com padrões não lineares.

O estudo aprofundou conceitos de arquitetura neural, treinamento, regularização e capacidade de generalização.

**Principais conceitos:** redes neurais, funções de ativação, normalização, Dropout, Batch Normalization, regularização L2, overfitting, AUC e threshold de classificação.

[Consultar estudo - Dataset Moon](https://userdanixdev.github.io/ml_mlops/moon/){ .md-button }

### 3. Redes neurais artificiais aplicadas à regressão

Aplicação de redes neurais com PyTorch em um cenário de estimativa de valores de seguros automotivos, utilizando dados sintéticos e registros com valores ausentes.

O experimento investiga a influência das funções de ativação e a capacidade de uma rede neural aprender relações entre variáveis para realizar previsões.

**Principais conceitos:** regressão, pesos e bias, funções de ativação, normalização, treinamento, MAE, RMSE e R².

[Consultar estudo - Seguros Automotivos](https://userdanixdev.github.io/ml_mlops/seguros/){ .md-button }

## Projeto MLops - Breast Cancer Wisconsin

### Do desenvolvimento do modelo à sua disponibilização

Nesta etapa, os estudos avançam para uma abordagem mais abrangente de engenharia de Machine Learning.

O objetivo é desenvolver um pipeline estruturado, contemplando não apenas a construção e avaliação de modelos, mas também práticas utilizadas na organização, rastreabilidade, disponibilização e monitoramento de soluções de Machine Learning.

Para isso, será utilizado o dataset **Breast Cancer Wisconsin**, disponibilizado pelo Scikit-learn, em um problema de classificação binária.

O desenvolvimento será organizado em etapas:

1. **Preparação e análise dos dados:** carregamento, exploração, verificação da qualidade e pré-processamento.
2. **Treinamento dos modelos:** implementação de modelos baseline e Random Forest.
3. **Otimização e rastreamento:** ajuste de hiperparâmetros e acompanhamento de experimentos com MLflow.
4. **Registro do modelo:** versionamento e gerenciamento dos modelos utilizando MLflow Model Registry.
5. **Monitoramento:** simulação de Data Drift e acompanhamento de métricas.
6. **API de inferência:** disponibilização do modelo por meio de uma API desenvolvida com FastAPI.
7. **Containerização:** empacotamento da aplicação utilizando Docker.
8. **Orquestração:** implantação e gerenciamento dos containers com Kubernetes.

A evolução dos estudos busca construir uma compreensão progressiva do ciclo de desenvolvimento de Machine Learning.

A partir dos fundamentos de classificação e regressão, passando pela construção e avaliação de redes neurais, o portfólio avança para práticas de MLOps, aproximando o desenvolvimento experimental de um fluxo estruturado de engenharia e disponibilização de modelos.

**A proposta é aprender não somente a desenvolver um modelo que realiza previsões, mas também compreender como organizar, avaliar, rastrear, disponibilizar e acompanhar seu funcionamento ao longo do ciclo de vida.**

---

**Autor:** Daniel Martins França

**Repositório:** [github.com/userdanixdev/ml_mlops](https://github.com/userdanixdev/ml_mlops/tree/project_mlops){ .md-button }