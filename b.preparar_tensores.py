import pandas as pd
import torch

df = pd.read_csv("src/bronze/dataset_seguros_normalizado.csv")
print(df.head())

# Modificação das variáveis: Tensionar em Vetores

features = ["idade_normalizada"]
target = "seguro_normalizado"    

# Registros com valores preenchidos para treinamento
df_treino = df.dropna(subset=[target]).copy()
# Registros com valores ausentes que serão usados para previsão
df_prever = df[df[target].isna()].copy()

print("Registros para treinamento: ",len(df_treino))
print("Registros para previsão: ",len(df_prever))

# Conversão dos dados em tensores ( vetores ):

X = torch.tensor(
    df_treino[features].values,
    dtype=torch.float32
)
y = torch.tensor(
    df_treino[target].values,
    dtype=torch.float32
).reshape(-1,1)

# Preparar as entradas que receberão previsões:

X_prever = torch.tensor(
    df_prever[features].values,
    dtype=torch.float32
)

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Formato de X_prever:\n", X_prever.shape)
