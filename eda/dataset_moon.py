
import pandas as pd
from sklearn.datasets import make_moons
from pathlib import Path

# Reprodutibilidade
SEED = 42

# Gerar o dataset
X, y = make_moons(
    n_samples=1000,
    noise=0.20,
    random_state=SEED
)

colunas = []
for i in range(X.shape[1]):
    nome_coluna = f"x{i+1}"
    colunas.append(nome_coluna)
print(colunas)    

# Criar DataFrame
df = pd.DataFrame(X, columns=colunas)
# Acrescentar coluna classe alvo:
df["classe"] = y

# Visualizar os dados
print("\n --- Primeiras Observações ---")
print(df.head(10))
print("\n --- Últimas Observações ---")
print(df.tail())
print("\nDimensões:", df.shape)
print("\nColunas:")
print(df.columns.tolist())
print("\nTipos de dados:")
print(df.dtypes)
print("\nValores únicos por coluna:")
print(df.nunique())
print("\nDistribuição das classes:")
print(df["classe"].value_counts().sort_index())
print("\nValores ausentes:")
print(df.isna().sum())
print("\nEstatísticas descritivas:")
print(df.describe())

# Definição do caminho:
pasta_data = Path(__file__).resolve().parent.parent / "data"
# Salvar o DataFrame em CSV:
path_csv = pasta_data / "dataset_moon.csv"
df.to_csv(path_csv, index=False)
print(f"\n Dataset salvo em: {path_csv}")
