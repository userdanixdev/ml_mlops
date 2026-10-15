import numpy as np
import pandas as pd

# Reprodutibilidade dos dados

np.random.seed(42)

# Quantidade de registros:

N= 100

# Nomes fictícios:
nomes = [
    f"Segurado_{i:03d}"
    for i in range(1,N+1)
]

# Idade entre 18 e 80 anos

idades = np.random.randint(18,81,size=N)

# Distância da idade em relação a 40 anos:
# Cria uma relação em U ente idade e valor do seguro

efeito_idade = (idades - 40) **2

# Valor sintético do seguro em reais:
valores_seguro = (
    1200
    + 1.5 * efeito_idade
    + np.random.normal(0,250,size=N)
)
# Evitar valores negativos
valores_seguro = np.maximum(valores_seguro,500)

# Dataset:
df = pd.DataFrame({
    "nome":nomes,
    "idade":idades,
    "valor_seguro":np.round(valores_seguro,2)
})

# valores nulos:
nulos = np.random.choice(
    df.index,size=15,
    replace = False
)
df.loc[nulos,"valor_seguro"] = np.nan

# Salvar o dataset:

df.to_csv(
    "src/dataset_seguros_vertice.csv",
    index = False
)
print(df.to_string(index=False))
print("\nResumo do dataset:")
print(df.info())

print("\n Valores ausentes:")
print(df.isnull().sum())

print("\nArquivo salvo: dataset_seguros_vertice.csv")