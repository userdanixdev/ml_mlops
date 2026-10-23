# Preparação dos dados - Dataset Moon

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_dados():
    root = Path(__file__).resolve().parents[2]
    path_csv = root / "data" / "dataset_moon.csv"

    df = pd.read_csv(path_csv)

    return df


def preparar_dados(df):
    # Separação das características e do alvo
    X = df.drop(columns=["classe"])
    y = df["classe"]

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Total de registros: {len(df)}")
    print(f"X_train: {len(X_train)}")
    print(f"y_train: {len(y_train)}")
    print(f"X_test: {len(X_test)}")
    print(f"y_test: {len(y_test)}")

    # Padronização das características
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
