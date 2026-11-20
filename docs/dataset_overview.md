# Dataset Breast Cancer Wisconsin

O carregamento do dataset foi realizado utilizando a biblioteca **Scikit-Learn**, que disponibiliza o conjunto de dados `Breast Cancer Wisconsin` por meio da função `load_breast_cancer().`

### Carregamento do dataset
```python
def load_dataset():
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    return df
```

A função `load_breast_cancer(as_frame=True)` carrega os dados diretamente do `Scikit-Learn` em estruturas compatíveis com o *Pandas*.

O atributo `dataset.frame` reúne as 30 features e a variável target em um único DataFrame. O método `copy()` cria uma cópia dos dados para trabalhar sem modificar a estrutura original.

### Geração do relatório

```python
def generate_dataset_report(data):
    report_path = Path("reports/dataset_overview.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
```    

A função `generate_dataset_report()` recebe o DataFrame e gera automaticamente um relatório em **Markdown.**

O relatório apresenta informações sobre a quantidade de registros e colunas, nome e tipo de dado das colunas, quantidade de valores ausentes e quantidade de registros duplicados.

A verificação da qualidade dos dados é realizada com:

```python
missing_values = int(data.isnull().sum().sum())
duplicated = int(data.duplicated().sum())
```

> Essas operações permitem identificar valores ausentes e registros duplicados no conjunto de dados.

## Características do dataset

| Característica | Descrição |
|---|---|
| Registros | 569 amostras |
| Features | 30 variáveis numéricas |
| Target | 1 variável de classificação |
| Problema | Classificação binária |
| Valores ausentes | Nenhum |
| Origem | Breast Cancer Wisconsin (Diagnostic) |

As 30 features representam características extraídas de imagens digitalizadas de massas mamárias, como:

- **Raio (radius):** distância média do centro até o perímetro.
- **Textura (texture):** variação dos níveis de escala de cinza.
- **Perímetro (perimeter):** medida do contorno da massa.
- **Área (area):** tamanho da região delimitada.
- **Suavidade (smoothness):** variação local dos comprimentos do raio.
- **Compacidade (compactness):** relação entre perímetro e área.
- **Concavidade (concavity):** gravidade das porções côncavas do contorno.
- **Pontos côncavos (concave points):** quantidade de pontos côncavos no contorno.
- **Simetria (symmetry):** grau de simetria da massa.
- **Dimensão fractal (fractal dimension):** medida relacionada à complexidade do contorno.

**Essas características são disponibilizadas em três grupos: valores médios (`mean`), erros padrão (`error`) e valores extremos (`worst`), totalizando 30 variáveis.**

## Variável target e classificação

A variável `target` representa o diagnóstico da amostra:

| Valor | Classe | Significado |
|---|---|---|
| 0 | Malignant | Tumor maligno |
| 1 | Benign | Tumor benigno |

A distribuição das classes é:

- 212 amostras malignas.
- 357 amostras benignas.

*Essa distribuição mostra que as classes não possuem a mesma quantidade de registros, algo importante para considerar durante o treinamento e a avaliação dos modelos.*

## Identificação
    - Dataset: Breast Cancer Wisconsin
    - Origem: Scikit-learn
## Dimensões
    - Registros: 569        
    - Colunas: 31
## Estrutura das colunas
| Coluna | Tipo de dado |
|---|---|
| mean radius | float64 |
| mean texture | float64 |
| mean perimeter | float64 |
| mean area | float64 |
| mean smoothness | float64 |
| mean compactness | float64 |
| mean concavity | float64 |
| mean concave points | float64 |
| mean symmetry | float64 |
| mean fractal dimension | float64 |
| radius error | float64 |
| texture error | float64 |
| perimeter error | float64 |
| area error | float64 |
| smoothness error | float64 |
| compactness error | float64 |
| concavity error | float64 |
| concave points error | float64 |
| symmetry error | float64 |
| fractal dimension error | float64 |
| worst radius | float64 |
| worst texture | float64 |
| worst perimeter | float64 |
| worst area | float64 |
| worst smoothness | float64 |
| worst compactness | float64 |
| worst concavity | float64 |
| worst concave points | float64 |
| worst symmetry | float64 |
| worst fractal dimension | float64 |
| target | int64 |

## Qualidade dos dados
- Valores ausentes: 0
- Registros Duplicados: 0

## O que acontece com o target no pré-processamento?

O target não deve passar pela mesma normalização ou padronização das features.

Ele já está codificado numericamente como 0 e 1. 

Portanto:

- **X**: poderá receber padronização ou normalização.
- **y**: permanece com seus valores originais (0 e 1).
- **train_test_split**: divide ambos, mantendo a correspondência entre as amostras.

Os scripts estão modularizados e podem ser conferidos aqui:

[Documentação Técnica - Load ](https://github.com/userdanixdev/ml_mlops/blob/project_mlops/src/mlops/data/load_dataset.py){.md-button }