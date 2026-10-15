import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

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
    "Spending_Score": spending_score
})

# Adicionando Outliers:

outliers = pd.DataFrame({
    "Annual_Income":[21,23,25,90,94],
    "Spending_Score":[2,3,100,100,2]
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
df_scaled = pd.DataFrame(df_scaled, columns=['Annual_Income',"Spending_Score"])
print(df_scaled.head())

dbscan = DBSCAN(
    eps=0.5,
    min_samples=3
    )
dbscan.fit(df_scaled)
df_scaled['Cluster']=dbscan.labels_
print(df_scaled.head())

plt.figure(figsize=(10, 6))

plt.scatter(
    df_scaled["Annual_Income"],
    df_scaled["Spending_Score"],
    c=df_scaled["Cluster"]
)

plt.xlabel("Annual Income (milhares)")
plt.ylabel("Spending Score")
plt.title("Clusters identificados pelo K-Means")

plt.show()




