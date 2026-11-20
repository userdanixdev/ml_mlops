# Relatório de Pré-Processamento

O pré-processamento é uma etapa importante do pipeline de **Machine Learning**, pois organiza os dados e verifica suas condições antes do treinamento dos modelos.

Neste projeto, essa etapa foi desenvolvida para separar as variáveis preditoras, verificar a qualidade e as escalas dos dados e dividir o conjunto em treinamento e teste.

## 1. Separação das variáveis

A função `separate_features_target()` separa as 30 variáveis preditoras (`X`) da variável `target` (`y`), que representa a classificação dos tumores.

```python
X = data.drop(columns=["target"])
y = data["target"]
```

> Essa separação permite que o modelo utilize as características das amostras para aprender a identificar as classes.

## 2. Verificação dos dados

A função `check_transformations()` realiza verificações antes do treinamento, identificando valores ausentes, tipos numéricos, quantidade de features e diferenças entre as escalas das variáveis.

Essa análise é importante porque as `features` possuem amplitudes diferentes. Dependendo do algoritmo utilizado, poderá ser necessário aplicar técnicas de padronização ou normalização.

## 3. Divisão entre treinamento e teste

Os dados são divididos utilizando a função `train_test_split()` do `Scikit-learn`, separando **80%** para treinamento e **20%** para teste.

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- `test_size=0.2`: reserva 20% dos registros para teste.
- `random_state=42`: garante a reprodutibilidade da divisão.
- `stratify=y`: mantém a proporção das classes nos dois conjuntos.

## 4. Relatório automatizado

A função `generate_preprocessing_report()` gera automaticamente o arquivo `reports/preprocessing_report.md`, contendo informações sobre a separação das variáveis, verificações realizadas, escalas e dimensões dos conjuntos de treinamento e teste.

Ao final dessa etapa, os dados estão organizados e divididos para utilização no treinamento e na avaliação dos modelos.

**Observação:** nesta fase são realizadas verificações e a divisão dos dados. A padronização ou normalização das features ainda não foi aplicada.

## Separação das variáveis

- Features: 30    
- Registros: 569
- Variável target: target

## Verificação das features

- Valores Ausentes: 0
- Variáveis numéricas: 30
- Quantidade de features: 30

## Escalas das variáveis

| Feature | Valor mínimo | Valor Máximo |
|---|---|---:|
| mean radius | 6.9810 | 28.1100 |
| mean texture | 9.7100 | 39.2800 |
| mean perimeter | 43.7900 | 188.5000 |
| mean area | 143.5000 | 2501.0000 |
| mean smoothness | 0.0526 | 0.1634 |
| mean compactness | 0.0194 | 0.3454 |
| mean concavity | 0.0000 | 0.4268 |
| mean concave points | 0.0000 | 0.2012 |
| mean symmetry | 0.1060 | 0.3040 |
| mean fractal dimension | 0.0500 | 0.0974 |
| radius error | 0.1115 | 2.8730 |
| texture error | 0.3602 | 4.8850 |
| perimeter error | 0.7570 | 21.9800 |
| area error | 6.8020 | 542.2000 |
| smoothness error | 0.0017 | 0.0311 |
| compactness error | 0.0023 | 0.1354 |
| concavity error | 0.0000 | 0.3960 |
| concave points error | 0.0000 | 0.0528 |
| symmetry error | 0.0079 | 0.0790 |
| fractal dimension error | 0.0009 | 0.0298 |
| worst radius | 7.9300 | 36.0400 |
| worst texture | 12.0200 | 49.5400 |
| worst perimeter | 50.4100 | 251.2000 |
| worst area | 185.2000 | 4254.0000 |
| worst smoothness | 0.0712 | 0.2226 |
| worst compactness | 0.0273 | 1.0580 |
| worst concavity | 0.0000 | 1.2520 |
| worst concave points | 0.0000 | 0.2910 |
| worst symmetry | 0.1565 | 0.6638 |
| worst fractal dimension | 0.0550 | 0.2075 |

## Divisão dos dados

- Proporção de treinamento: 80%
- Proporção de teste: 20%
- Random state: 42
- Estratificação: Sim

## Dimensões dos conjuntos

| Conjunto | Registros | Features |
|---|---:|---:|
| X_train | 455 | 30 |
| X_test | 114 | 30 |
| y_train | 455 | - |
| y_test | 114 | - |

Caso queria consultar a documentação técnica completa da fase de pré-processamento:

[Documentação Técnica Pré-Processamento](https://github.com/userdanixdev/ml_mlops/blob/project_mlops/src/mlops/preprocessing/pre_process.py){ .md-button}      
