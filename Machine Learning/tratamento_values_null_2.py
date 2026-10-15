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

# Utilizando o SimpleImputer:

df_fill = df_null.copy() # Crie uma cópia do dataframe
num_imputer = SimpleImputer(strategy='median') # Parâmetro = Mediana
filled = num_imputer.fit_transform(df_fill) # 
df_imputed = pd.DataFrame(filled, columns=df_fill.columns)
print(f"\n Verificação de tratamento de dados com Simple Imputer - Pipelines: \n")
for coluna in df_imputed.columns:
    quant_null = df_imputed[coluna].isnull().sum()
    if quant_null == 0:
        print(f" Coluna {coluna}: OK - Nenhum valor nulo")
    else:
        print(f" Coluna{coluna}: {quant_null} valores nulos encontrados.")                
