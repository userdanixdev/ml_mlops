# Arquitetura da rede - Dataset Moon

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path_csv = root / "data" / "dataset_moon.csv"

# Carregar o dataframe criado na etapa EDA:
df = pd.read_csv(path_csv)

# Separação das variáveis: 'X' para as características e alvo 'y'

X = df.drop(columns=["classe"])
y = df["classe"]

# Divisão de dados para treino e teste:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,stratify=y)

print(f"Total de registros do DataFrame: {len(df)}")
print(f"Dados de treino ( X_Train): {len(X_train)}")
print(f"Dados de teste: ( X_test ){len(X_test)}")
print(f"Dados de treino 'y_train' {len(y_train)}")
print(f"Dados de test 'y_test' {len(y_test)}")


# Padronização:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Arquitetura inicial do Modelo MLP:
inputs = keras.Input(shape=(X_train.shape[1],))

x= layers.Dense(16, activation="relu")(inputs)
##x = layers.BatchNormalization()(x)
##x = layers.Dropout(0.2)(x)
x = layers.Dense(16, activation="relu")(x)
#x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1,activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
# Exibição da arquitetura:
model.summary()

# Por enquanto, deixamos L2, Batch Normalization e Dropout de fora. 
# Essa será nossa arquitetura de referência, permitindo comparar os resultados quando adicionarmos
# cada técnica.
