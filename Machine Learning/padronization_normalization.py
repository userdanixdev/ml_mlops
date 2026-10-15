import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def preparar_dados():
    np.random.seed(42)

    data_scale = np.array([
        np.random.randint(10**i,10**(i+1),10000, dtype=np.int64)
        for i in range(10)]).T

    print(f"\n Dimensões do Array: {data_scale.shape}\n")
    print(f"\n As primeiras 5 linhas: {data_scale[:5]}\n\n")

    # Data Frame:
    df_scale = pd.DataFrame(data_scale,columns=[f'feature_{i}' for i in range(1,11)])
    print(df_scale.head())

    # Padronização com StandardScaler do SKTlearn:

    dados_padronizados = data_scale.copy()
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados_padronizados).round(3)
    print(f"\nDados após padronização:\n{dados_padronizados[:5]}\n")


    minmax = MinMaxScaler()
    dados_normalizados = minmax.fit_transform(dados_padronizados).round(3)
    print(f"\n Dados Normalizados (0-1): \n{dados_normalizados[:5]}\n")
    print(f"\nPara encontrar o valor mínimo e máximo de cada coluna temos:\n")
    # Mínimo e máximo após normalização
    print("\nMínimo de cada coluna após normalização:")
    print(dados_normalizados.min(axis=0))

    print("\nMáximo de cada coluna após normalização:")
    print(dados_normalizados.max(axis=0))

    # Visualizando pelo DataFrame:

    df_padronizado = pd.DataFrame(
        dados_padronizados,
        columns=df_scale.columns
    )

    df_normalizado = pd.DataFrame(
        dados_normalizados,
        columns=df_scale.columns
    )

    print("\nDataFrame Padronizado:")
    print(df_padronizado.head())

    print("\nDataFrame Normalizado:")
    print(df_normalizado.head())

    print("\nMínimo Normalizado:")
    print(df_normalizado.min())

    print("\nMáximo Normalizado:")
    print(df_normalizado.max())

    return dados_normalizados

