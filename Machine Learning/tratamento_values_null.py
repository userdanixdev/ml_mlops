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

np.random.seed(42)

df_null = pd.DataFrame(
    np.random.randint(1, 1000, size=(1000, 10)),
    columns=[f'col{i}' for i in range(1, 11)]
)

# Coloca aproximadamente 15% de valores nulos
mask = np.random.random(df_null.shape) < 0.15
df_null[mask] = np.nan

print(f"Dimensões do Data Frame: {df_null.shape}")
print(f"\n Primeiras 5 linhas do DataFrame criado: \n :{df_null.head()}")
print(f"\n Quantidade de valores nulos por coluna: \n{df_null.isnull().sum()}")
print(f"\n Quantidade total de valores nulos: {df_null.isnull().sum().sum()}")

# Tratamento de valores nulos:
df_fill = df_null.copy() # Crie uma cópia do dataframe
media_col1 = df_fill['col1'].mean() # Calcule a média de valores da coluna 1
df_fill['col1']= df_fill['col1'].fillna(media_col1) # Preencha com os valores nulos com a média
print(f"\n\n Tratamentos de nulos da coluna 1 após preenchimento com a média (pandas): \n {df_fill[['col1']]}")
print(f"\n Quantidade total de valores nulos atual: {df_fill['col1'].isnull().sum().sum()}")

for coluna in df_fill.columns:
    media = df_fill[coluna].mean()
    print(f"Tratamento a coluna: {coluna}")
    print(f"Média calculada: {media:.2f}")
    df_fill[coluna] = df_fill[coluna].fillna(media)

print(f"\nQuantidade total de valores nulos após o tratamento: "
      f"{df_fill.isnull().sum().sum()}")
