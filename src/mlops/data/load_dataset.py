from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

def generate_dataset_report(data):
    report_path = Path("reports/dataset_overview.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    columns_table = "\n".join(
        f"| {column} | {data[column].dtype} |"
        for column in data.columns
    )
    missing_values = int(data.isnull().sum().sum())
    duplicated = int(data.duplicated().sum())

    report = f"""# Visão Geral do Dataset
## Identificação
    - Dataset: Breast Cancer Wisconsin
    - Origem: Scikit-learn
## Dimensões
    - Registros: {data.shape[0]}        
    - Colunas: {data.shape[1]}
## Estrutura das colunas
| Coluna | Tipo de dado |
|---|---|
{columns_table}

## Qualidade dos dados
- Valores ausentes: {missing_values}
- Registros Duplicados: {duplicated}
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"\nRelatório gerado em: {report_path}")

def load_dataset():
    """
    Carregamento do dataset Breast Cancer Wisconsin
    Retorna dataframe com os dados do dataset
    """
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    
    return df

if __name__ == "__main__":
    data = load_dataset()
    generate_dataset_report(data)

    print("Dataset Breast Cancer Wisconsin")
    print("-"*40)
    print(f"Quantidade de registros: {data.shape[0]}")
    print(f"Quantidade de colunas: {data.shape[1]}")
    print("\nNomes das colunas:")
    for coluna in data.columns:
        print(coluna)
    print("\nTipos de dados:")        
    print(data.dtypes)
    print("\n Primeiras linhas:")
    print(data.head())
    print("\nQuantidade de valores ausentes por coluna:")
    print(data.isnull().sum())
    print("\nTotal de valores ausentes:")
    print(data.isnull().sum().sum())
    print("\nQuantidade de registros duplicados:")
    print(data.duplicated().sum())
    print("\nEstatísticas descritivas:")
    print(data.describe().T)