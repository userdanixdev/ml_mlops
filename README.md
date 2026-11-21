![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Management-60A5FA?logo=poetry&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5?logo=kubernetes&logoColor=white)




# Projeto MLops - Breast Cancer Wisconsin

## Visão Geral

Este projeto tem como objetivo desenvolver um pipeline completo de Machine Learning, aplicando práticas de MLOps desde a preparação dos dados até a disponibilização e o monitoramento de um modelo em ambiente de produção.

Será utilizado o dataset ```Breast Cancer Wisconsin```, disponibilizado pelo **Scikit-learn**, para desenvolver modelos capazes de classificar tumores como benignos ou malignos.

O projeto utilizará `Python` e ferramentas voltadas à análise de dados, treinamento, rastreamento de experimentos, disponibilização de modelos e infraestrutura.

## Tecnologias e Ferramentas

- **Python**: linguagem utilizada no desenvolvimento.
- **Poetry**: gerenciamento de dependências e ambiente virtual.
- **Scikit-learn**: preparação dos dados, treinamento e avaliação dos modelos.
- **Pandas** : manipulação e operações numéricas.
- **Matplotlib** e **Seaborn**: visualização de dados.
- **MLflow**: rastreamento de experimentos e gerenciamento de modelos.
- **FastAPI**: desenvolvimento da API de inferência.
- **Pydantic**: validação dos dados de entrada.
- **Uvicorn**: servidor ASGI para execução da API.
- **Docker**: containerização da aplicação.
- **Kubernetes**: orquestração dos containers.


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

### Comandos utilitários:

- Instalação das dependências principais:

```bash
poetry add numpy pandas scikit-learn matplotlib seaborn mlflow fastapi "uvicorn[standard]" pydantic
```

## Arquitetura:

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
