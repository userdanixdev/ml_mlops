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


# Codificação de Variáveis Categóricas:
## Label Encoding (ordinal)
# Use quando houver ordem natural entre as categorias (ex.:P<M<G)

df_category = pd.DataFrame({'produtos':['camiseta','calça','jaqueta','bermuda','calça','camiseta','jaqueta'],
                            'tamanho':['P','G','M','G','GG','PP','P']})
print(df_category)

label_encoder = LabelEncoder()
df_category['tamanho_encoded'] = label_encoder.fit_transform(df_category['tamanho'])
print(df_category)
