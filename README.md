![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit-Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

# Classificação Binária de Desempenho de Alunos com PyTorch

## Sobre o projeto

Este projeto foi desenvolvido com o objetivo de aplicar conceitos de **Machine Learning e Deep Learning com Python e PyTorch**, construindo um fluxo completo de classificação binária para prever o resultado acadêmico de estudantes.

A aplicação utiliza um dataset sintético gerado com NumPy, contendo características relacionadas ao estudo, conhecimento prévio e facilidade de aprendizado. A partir desses dados, um modelo de classificação é treinado para identificar padrões e prever resultados ausentes.

## Objetivos

* Preparar os dados para treinamento e previsão.
* Implementar e treinar um modelo de classificação binária com PyTorch.
* Prever resultados ausentes sem modificar o arquivo original.
* Avaliar o modelo utilizando métricas de classificação e matriz de confusão.

## Tecnologias utilizadas

* **Python** — linguagem de programação.
* **NumPy** — geração e manipulação de dados numéricos.
* **Pandas** — tratamento e análise de dados tabulares.
* **PyTorch** — construção e treinamento do modelo.
* **Scikit-learn** — divisão dos dados e avaliação do modelo.
* **OpenPyXL** — suporte à leitura e gravação de arquivos Excel.

## Estrutura do projeto

```text
Deep Learning/
│
├── dp_learning_classification_1.py
├── dp_learning_classification_2.py
├── dp_learning_classification_3.py
├── dp_learning_classification_4.py
├── dp_learning_classification_5.py
├── dp_learning_classification_6.py
│
├── alunos_base_original.xlsx
├── alunos_com_predicoes.xlsx
└── README.md
```

### Responsabilidade dos arquivos

| Arquivo                           | Responsabilidade                                                    |
| --------------------------------- | ------------------------------------------------------------------- |
| `dp_learning_classification_1.py` | Geração do dataset simples inicial para estudos.
| `dp_learning_classification_2.py` | Geração de uma dataset robusto como base e preparação dos dados para treinamento e previsão com 10.000 registros. |
| `dp_learning_classification_3.py` | Construção e treinamento do modelo de classificação com PyTorch.    |
| `dp_learning_classification_4.py` | Geração das previsões e exportação para um novo arquivo Excel.      |
| `dp_learning_classification_5.py` | Verificação da integridade dos dados e das previsões exportadas.    |
| `dp_learning_classification_6.py` | Avaliação do modelo utilizando dados separados para teste.          |

## Dataset

A base sintética contém **10.000 registros**, com três variáveis preditoras e uma variável-alvo.

| Variável                 | Descrição                                                    |
| ------------------------ | ------------------------------------------------------------ |
| `nome`                   | Identificação do estudante.                                  |
| `estudou`                | Indicador numérico de estudo.                                |
| `conhecimento_previo`    | Indicador de conhecimento prévio.                            |
| `facilidade_aprendizado` | Indicador de facilidade de aprendizado.                      |
| `resultado`              | Resultado conhecido: `1` para aprovado e `0` para reprovado. |

Para simular um cenário de previsão, 20% dos resultados foram inicialmente deixados em branco.

Assim, a base contém aproximadamente 8.000 registros com resultados conhecidos e 2.000 registros destinados à previsão.

## Desenvolvimento do modelo

O modelo utiliza uma camada linear do PyTorch com três entradas e uma saída:

```python
nn.Linear(
    in_features=3,
    out_features=1,
    bias=True
)
```

A saída representa um *logit*, convertido em probabilidade por meio da função sigmoide.

### Configurações de treinamento

* **Arquitetura:** camada linear com três entradas e uma saída.
* **Função de perda:** `BCEWithLogitsLoss`.
* **Otimizador:** Adam.
* **Taxa de aprendizado:** `0.01`.
* **Épocas:** `1000`.
* **Critério de classificação:** probabilidade igual ou superior a `0.5` resulta em classe `1`; abaixo desse limite, classe `0`.

A arquitetura implementada corresponde a um modelo linear de classificação binária, semelhante à regressão logística, utilizada aqui como etapa introdutória ao desenvolvimento de modelos com PyTorch.

## Avaliação do modelo

Para avaliar a capacidade de generalização, os 8.000 registros com resultados conhecidos foram divididos em:

* **Treinamento:** 6.400 registros (80%).
* **Teste:** 1.600 registros (20%).

A avaliação foi realizada com o Scikit-learn.

### Métricas obtidas

| Métrica                   | Resultado |
| ------------------------- | --------: |
| Accuracy                  |    96,25% |
| Precision                 |    93,77% |
| Recall                    |    99,27% |
| F1-score                  |    96,44% |
| Loss final de treinamento |    0,2431 |

### Matriz de confusão

```text
[[727  54]
 [  6 813]]
```

Considerando a classe `1` como aprovação:

* **727 verdadeiros negativos:** alunos reprovados classificados corretamente.
* **54 falsos positivos:** alunos reprovados classificados incorretamente como aprovados.
* **6 falsos negativos:** alunos aprovados classificados incorretamente como reprovados.
* **813 verdadeiros positivos:** alunos aprovados classificados corretamente.

Os resultados demonstram bom desempenho no conjunto de teste utilizado neste experimento.

## Arquivos de saída

### `alunos_base_original.xlsx`

Contém os dados originais gerados pelo projeto, incluindo os resultados conhecidos e os campos inicialmente vazios.

### `alunos_com_predicoes.xlsx`

Contém os dados originais e as colunas adicionais:

* `resultado_predito`: resultado original, quando conhecido, ou previsão do modelo para os registros inicialmente sem resultado.
* `probabilidade_predicao`: probabilidade produzida pelo modelo para os registros previstos.

A base original é preservada, permitindo distinguir os resultados conhecidos das previsões geradas.

### Fluxo do estudo:

[](/images/Fluxo%20Completo%20de%20Classificação%20Acadêmica.png)

## Como executar

### 1. Instalar as dependências

```bash
python -m pip install numpy pandas torch scikit-learn openpyxl
```

### 2. Executar os arquivos na sequência

```bash
python dp_learning_classification_1.py
python dp_learning_classification_2.py
python dp_learning_classification_3.py
python dp_learning_classification_4.py
python dp_learning_classification_5.py
python dp_learning_classification_6.py
```

**Observação:** os arquivos devem estar na mesma pasta. Como a etapa 4 importa o código da etapa 3, ela executa novamente o treinamento ao ser iniciada. A etapa 6 realiza um treinamento independente para avaliar o modelo, utilizando sua própria divisão entre treinamento e teste.

## Observações:

O dataset utilizado é sintético, e o resultado original foi gerado a partir de uma regra matemática que utiliza as mesmas variáveis preditoras fornecidas ao modelo.

Por esse motivo, as métricas obtidas representam o desempenho neste experimento controlado e não garantem resultados equivalentes em dados acadêmicos reais.

Além disso, as probabilidades produzidas pelo modelo não devem ser interpretadas automaticamente como probabilidades perfeitamente calibradas.

## Aprendizados

O desenvolvimento deste projeto permitiu praticar conceitos fundamentais de:

* Manipulação e preparação de dados.
* Conversão de arrays NumPy e DataFrames em tensores PyTorch.
* Construção e treinamento de modelos de classificação.
* Uso de funções de perda e otimizadores.
* Inferência com modelos treinados.
* Exportação e validação de resultados.
* Avaliação de modelos por meio de métricas e matriz de confusão.
* Organização de um fluxo de Machine Learning em múltiplos scripts Python.

## Conclusão

O projeto demonstra a implementação de um fluxo introdutório de classificação binária com Python e PyTorch, desde a geração dos dados até a avaliação do modelo.

A experiência consolida conceitos essenciais de preparação de dados, treinamento, inferência e validação, servindo como base para estudos posteriores em Machine Learning e Deep Learning.


