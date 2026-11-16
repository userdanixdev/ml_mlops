# Dataset Breast Cancer Wisconsin

O **Breast Cancer Wisconsin (Diagnostic)** é um conjunto de dados utilizado em estudos de Machine Learning para desenvolver modelos de **classificação binária**, com o objetivo de identificar se um tumor mamário apresenta características malignas ou benignas.

Neste projeto, utilizamos a versão disponibilizada pelo Scikit-learn, por meio da função `load_breast_cancer()`.

## Origem e documentação

O dataset foi desenvolvido a partir de pesquisas da Universidade de Wisconsin e está disponível no UCI Machine Learning Repository.

Documentações de referência:

- [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Scikit-learn – load_breast_cancer()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## Aplicação neste projeto

O dataset será utilizado para desenvolver um pipeline de Machine Learning voltado à classificação de tumores em malignos ou benignos.

A proposta é utilizar esse problema para explorar as etapas de um fluxo MLOps, desde o carregamento e preparação dos dados até o treinamento, avaliação, rastreamento de experimentos e disponibilização do modelo por meio de uma API.

**Observação:** este é um dataset educacional para experimentação com Machine Learning, não uma ferramenta para diagnóstico médico.

## Preparação e Análise dos Dados

**Utilizando o dataset o objetivo será preparar os dados para que possam ser utilizados no treinamento e na avaliação dos modelos de Machine Learning.**


### 1.Obtenção dos dados
   
   Carregar o Breast Cancer Wisconsin diretamente pelo Scikit-learn e compreender sua estrutura.

### 2.Análise exploratória 
   
   Investigar dimensões, tipos de dados, estatísticas descritivas e distribuição das classes.

### 3.Limpeza dos dados
   
   Verificar se possuem valores ausentes, duplicidades e possíveis inconsistências.

### 4.Pré-processamento
   
   Separar variáveis preditoras e variável-alvo, dividir os dados em treino e teste e aplicar normalização quando apropriado.



   
