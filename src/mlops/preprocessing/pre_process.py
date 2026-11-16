from sklearn.model_selection import train_test_split
from pathlib import Path

def separate_features_target(data):
    X = data.drop(columns=["target"])
    y = data["target"]

    return X,y

def check_transformations(X):
    print("\nVerificação para pré-processamento")
    print("-" * 40)

    print(f"Valores ausentes: {X.isnull().sum().sum()}")
    print(f"Tipos numéricos: {X.select_dtypes(include='number').shape[1]}")
    print(f"Quantidade de features: {X.shape[1]}")

    print("\nEscalas das variáveis:")
    print(X.describe().loc[["min", "max"]].T)


def train_test_split_data(X,y,test_size=0.2, random_state=42):
    return train_test_split(X,y,test_size=test_size, random_state=random_state,stratify=y)

def generate_preprocessing_report(X,y,X_train,X_test,y_train,Y_test)    :
    report_path = Path("reports/preprocessing_report.md")
    report_path.parent.mkdir(parents=True,exist_ok=True)

    scales_table = "\n".join(
        f"| {column} | {X[column].min():.4f} | {X[column].max():.4f} |"
        for column in X.columns
    )
    report = f"""# Relatório de Pré-Processamento
## Separação das variáveis

- Features: {X.shape[1]}    
- Registros: {X.shape[0]}
- Variável target: target

## Verificação das features

- Valores Ausentes: {int(X.isnull().sum().sum())}
- Variáveis numéricas: {X.select_dtypes(include="number").shape[1]}
- Quantidade de features: {X.shape[1]}

## Escalas das variáveis

| Feature | Valor mínimo | Valor Máximo |
|---|---|---:|
{scales_table}

## Divisão dos dados

- Proporção de treinamento: {X_train.shape[0] / X.shape[0]:.0%}
- Proporção de teste: {X_test.shape[0] / X.shape[0]:.0%}
- Random state: 42
- Estratificação: Sim

## Dimensões dos conjuntos

| Conjunto | Registros | Features |
|---|---:|---:|
| X_train | {X_train.shape[0]} | {X_train.shape[1]} |
| X_test | {X_test.shape[0]} | {X_test.shape[1]} |
| y_train | {y_train.shape[0]} | - |
| y_test | {y_test.shape[0]} | - |
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"\\nRelatório gerado em: {report_path}")


