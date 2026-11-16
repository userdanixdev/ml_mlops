# Preparação do Ambiente

## Configuração do ambiente e gerenciamento de dependências

O ambiente de desenvolvimento foi configurado utilizando **Python** e **Poetry**, ferramenta responsável pelo gerenciamento das dependências e pelo controle das versões das bibliotecas utilizadas no projeto.

O Poetry permite organizar as dependências principais e as ferramentas de desenvolvimento em grupos separados, facilitando a manutenção e a reprodução do ambiente.

### Dependências principais

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

### Dependências de desenvolvimento

| Biblioteca | Finalidade |
|---|---|
| Pytest | Testes automatizados |
| HTTPX | Testes de requisições e endpoints da API |
| Ruff | Análise estática e padronização do código |

### Comandos de instalação

**Instalação das dependências principais:**

```bash
poetry add numpy pandas scikit-learn matplotlib seaborn mlflow fastapi "uvicorn[standard]" pydantic
```

**Instalação das dependências de desenvolvimento:**

```bash
poetry add --group dev pytest httpx ruff
```

### Considerações

O Poetry mantém as configurações e versões das dependências nos arquivos `pyproject.toml` e `poetry.lock`, contribuindo para a organização e a reprodutibilidade do ambiente de desenvolvimento.