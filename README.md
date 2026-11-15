# Projeto MLops - Breast Cancer Wisconsin

## Visão Geral

Este projeto tem como objetivo desenvolver um pipeline completo de Machine Learning, aplicando práticas de MLOps desde a preparação dos dados até a disponibilização e o monitoramento de um modelo em ambiente de produção.

Será utilizado o dataset Breast Cancer Wisconsin, disponibilizado pelo Scikit-learn, para desenvolver modelos capazes de classificar tumores como benignos ou malignos.

O projeto utilizará Python e ferramentas voltadas à análise de dados, treinamento, rastreamento de experimentos, disponibilização de modelos e infraestrutura.

## Tecnologias e Ferramentas

- Python: linguagem utilizada no desenvolvimento.
- Poetry: gerenciamento de dependências e ambiente virtual.
- Scikit-learn: preparação dos dados, treinamento e avaliação dos modelos.
- Pandas e NumPy: manipulação e operações numéricas.
- Matplotlib e Seaborn: visualização de dados.
- MLflow: rastreamento de experimentos e gerenciamento de modelos.
- FastAPI: desenvolvimento da API de inferência.
- Pydantic: validação dos dados de entrada.
Uvicorn: servidor ASGI para execução da API.
- Docker: containerização da aplicação.
- Kubernetes: orquestração dos containers.
- Pytest, HTTPX e Ruff: testes automatizados e qualidade do código.

### Com Poetry, teremos:

- Gerenciamento centralizado das dependências.
- Controle de versões das bibliotecas.
- Arquivo poetry.lock para reproduzir o ambiente.
- Ambiente virtual isolado para o projeto.
- Instalação e atualização organizada dos pacotes.


## Pipeline MLOps

1. Preparação e análise dos dados
   
   Dataset Breast Cancer Wisconsin, exploração, limpeza e pré-processamento.
2. Treinamento dos modelos
   
   Random Forest e Regressão Logística como baseline.
3. Otimização e rastreamento
   
   RandomizedSearchCV, HalvingGridSearchCV e MLflow Tracking.
4. Registro do modelo
   
   MLflow Model Registry e versionamento do melhor modelo.
5. Monitoramento
   
   Simulação de Data Drift e registro de métricas.
6. API de inferência
   
   FastAPI ou Flask, disponibilizando o endpoint /predict.
7. Containerização
   
   Dockerfile, build e execução do container.
8. Orquestração
   
   Kubernetes, Deployment com duas réplicas, Service e testes de escalabilidade.


> O ambiente de desenvolvimento foi configurado com as bibliotecas necessárias para análise de dados, treinamento de modelos, rastreamento de experimentos, desenvolvimento da API e testes automatizados.

### Gerenciamento de dependências

O Poetry foi utilizado para gerenciar as bibliotecas do projeto, separando as dependências principais das ferramentas de desenvolvimento.

**Dependências principais:**

| Biblioteca | Finalidade |
|---|---|
| NumPy | Operações numéricas e computacionais |
| Pandas | Manipulação e análise de dados |
| Scikit-learn | Construção e avaliação de modelos de Machine Learning |
| Matplotlib | Visualização de dados |
| Seaborn | Visualização estatística |
| MLflow | Rastreamento de experimentos e gerenciamento de modelos |
| FastAPI | Desenvolvimento da API de predição |
| Uvicorn | Servidor ASGI para execução da API |
| Pydantic | Validação e estruturação dos dados de entrada |

**Dependências de desenvolvimento:**

| Biblioteca | Finalidade |
|---|---|
| Pytest | Testes automatizados |
| HTTPX | Testes de requisições e endpoints da API |
| Ruff | Análise estática e padronização do código |

### Comandos utilizados

Instalação das dependências principais:

```bash
poetry add numpy pandas scikit-learn matplotlib seaborn mlflow fastapi "uvicorn[standard]" pydantic
```

Instalação das dependências de desenvolvimento:

```bash
poetry add --group dev pytest httpx ruff
```

## Preparação e Análise dos Dados

Utilizando o dataset Breast Cancer Wisconsin, disponibilizado pelo Scikit-learn o objetivo será preparar os dados para que possam ser utilizados no treinamento e na avaliação dos modelos de Machine Learning.
O que vamos desenvolver
1. Obtenção dos dados
   
   Carregar o Breast Cancer Wisconsin diretamente pelo Scikit-learn e compreender sua estrutura.

2. Análise exploratória (EDA)
   
   Investigar dimensões, tipos de dados, estatísticas descritivas e distribuição das classes.
3. Limpeza dos dados
   
   Verificar se possue valores ausentes, duplicidades e possíveis inconsistências.

4. Pré-processamento
   
   Separar variáveis preditoras e variável-alvo, dividir os dados em treino e teste e aplicar normalização quando apropriado.

5. Documentação
   
   Registrar as análises, decisões e resultados no projeto.



```
ml_mlops/
│
├── src/
│   └── mlops/
│       ├── data/          # Carregamento e exploração dos dados
│       └── preprocessing/ # Limpeza e transformação
│
├── tests/                # Testes automatizados
├── reports/
├── docs/                 # Documentação
│
├── .gitignore
├── pyproject.toml
├── poetry.lock
└── README.md
```

O que acontece com o target no pré-processamento?
O target não deve passar pela mesma normalização ou padronização das features.

Ele já está codificado numericamente como 0 e 1. Portanto:
- X: poderá receber padronização ou normalização.
- y: permanece com seus valores originais (0 e 1).
- train_test_split: divide ambos, mantendo a correspondência entre as amostras.
Uma observação: sua função check_transformations(X) verifica somente as features. Isso é intencional, pois o target é tratado separadamente.   