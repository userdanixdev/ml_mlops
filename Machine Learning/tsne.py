import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Reprodutilibidade

np.random.seed(42)

# Grupo 1 : Baixa Renda + gasto crescente

n1 = 100
annual_income_1 = np.random.uniform(15,24.5,n1)
gasto_crescente = (
    39
    +(annual_income_1 - 15) * 6
    +np.random.normal(0,3,n1)
)
gasto_crescente = np.clip(
    gasto_crescente,
    1,
    100
)

# Grupo 02: Média Renda + Baixo custo
n2 = 100
annual_income_2 = np.random.uniform(25,29.5,n2)
gasto_crescente_2 = (
    6
    + (annual_income_2 - 25) * 5
    + np.random.normal(0,3,n2)
)
gasto_crescente_2 = np.clip(
    gasto_crescente_2,
    1,
    40
)
# Grupo 03: Alta Renda + Gasto Variável

n3 = 50
annual_income_3 = np.random.uniform(80,96,n3)
gasto_crescente_3 = np.random.uniform(
    40,
    90,
    n3
)

# Grupo de idades - Clientes mais jovens
age_1 = np.random.randint(18,36,n1)
# Grupo 2 - clientes de meia idade
age_2 = np.random.randint(30,51,n2)
# Grupo 3 - Cliente de maior idade:
age_3 = np.random.randint(40,71,n3)
age = np.concatenate([
    age_1,
    age_2,
    age_3
])

# Juntando os grupos:

annual_income = np.concatenate([
    annual_income_1,
    annual_income_2,
    annual_income_3
])
spending_score = np.concatenate([
    gasto_crescente,
    gasto_crescente_2,
    gasto_crescente_3
])
# criando o dataframe:

df = pd.DataFrame({
    "Annual_Income": annual_income,
    "Spending_Score": spending_score,
    "Age":age
})

# Adicionando Outliers:

outliers = pd.DataFrame({
    "Annual_Income":[21,23,25,90,94],
    "Spending_Score":[2,3,100,100,2],
    "Age": [19,22,31,68,65]
    })

df = pd.concat(
    [df,outliers],
    ignore_index= True
)

# Verificação
print(df.info())
print("\nNaN:")
print(df.isna().sum())

print("\nShape:")
print(df.shape)
print(df.head())

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=['Annual_Income',"Spending_Score","Age"])
print(df_scaled.head())

print("\nValores ausentes no DataFrame:")
print(df.isna().sum())

print("\nValores ausentes no DataFrame escalado:")
print(pd.isna(df_scaled).sum())

# Diminuir dimensionalidade TSNE:

tsne = TSNE(n_components=2, perplexity=10,random_state=42)
X_tsne = tsne.fit_transform(df_scaled)

df_tsne = pd.DataFrame(
    X_tsne,
    columns=["t-SNE1","t-SNE2"]
)
print(df_tsne.head())

# Visualização:

plt.figure(figsize=(10,6))
plt.scatter(
    df_tsne["t-SNE1"],
    df_tsne["t-SNE2"]
)
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.title("Redução de dimensionalidade com t-SNE")
plt.show()