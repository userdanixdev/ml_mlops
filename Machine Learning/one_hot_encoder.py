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
## One Hor enconding (nominal)
# Use quando houver sem ordem natural. Lidando com categorias desconhecidas em produção:

df_category_ohe = pd.DataFrame({'produtos':['camiseta','calça','jaqueta','bermuda','calça','camiseta','jaqueta'],
                            'cores':['vermelho','cinza','preta','preta','preta','azul','branca']})
print(df_category_ohe)

# 1º Forma de realizar One Hot Enconding via Pandas:
## Usando a função get_dummers: Converte variáveis categóricas em variáveis indicadoras ou binárias.

df_ohe_pd = pd.get_dummies(df_category_ohe, columns=['cores'], prefix='cor')
print(f"\nCom pandas usando a função '.get_dummies(): \n{df_ohe_pd}")

# 2 Forma:
# OneHotEncoder do Scikit Learn ( para pipelines ):

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df_category_ohe[['cores']])
df_ohe_skl = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['cores']))
print(f"\n Com OneHotEncoder do SKT:\n {df_ohe_skl}")