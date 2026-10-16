
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

# Carregar dataset
df = pd.read_csv("src/dataset_seguros_vertice.csv")

# Preservar os dados originais
df_scale = df.copy()

# Separar registros com seguro conhecido
df_conhecidos = df_scale[df_scale["valor_seguro"].notna()].copy()

# Variável de entrada e alvo conhecidos
X = df_scale[["idade"]]
y = df_conhecidos[["valor_seguro"]]

# Padronização das idades: todas as linhas
scaler_x = StandardScaler()
idades_padronizadas = scaler_x.fit_transform(X).ravel().round(3)

# Padronização dos seguros: somente valores conhecidos
scaler_y = StandardScaler()
seguros_padronizados = (
    scaler_y.fit_transform(y).ravel().round(3)
)

# Normalização das idades: todas as linhas
minmax_x = MinMaxScaler()
idades_normalizadas = minmax_x.fit_transform(X).ravel().round(3)

# Normalização dos seguros: somente valores conhecidos
minmax_y = MinMaxScaler()
seguros_normalizados = (
    minmax_y.fit_transform(y).ravel().round(3)
)

# Criar DataFrame padronizado completo
df_padronizado_completo = df_scale[
    ["nome", "idade", "valor_seguro"]
].copy()

df_padronizado_completo["idade_padronizada"] = idades_padronizadas
df_padronizado_completo["seguro_padronizado"] = float("nan")

df_padronizado_completo.loc[
    df_conhecidos.index, "seguro_padronizado"
] = seguros_padronizados

# Criar DataFrame normalizado completo
df_normalizado_completo = df_scale[
    ["nome", "idade", "valor_seguro"]
].copy()

df_normalizado_completo["idade_normalizada"] = idades_normalizadas
df_normalizado_completo["seguro_normalizado"] = float("nan")

df_normalizado_completo.loc[
    df_conhecidos.index, "seguro_normalizado"
] = seguros_normalizados

# Visualizar resultados
print("\nDados originais:")
print(df_scale.head(10))

print("\nDados padronizados:")
print(df_padronizado_completo.head(10))

print("\nDados normalizados:")
print(df_normalizado_completo.head(10))

print("\nMínimos após normalização:")
print(df_normalizado_completo[["idade_normalizada", "seguro_normalizado"]].min())

print("\nMáximos após normalização:")
print(df_normalizado_completo[["idade_normalizada", "seguro_normalizado"]].max())

print("\nValores ausentes:")
print(df_normalizado_completo.isna().sum())

# Criar diretório Bronze
pasta_bronze = Path("src") / "bronze"
pasta_bronze.mkdir(parents=True, exist_ok=True)

# Salvar datasets
arquivo_padronizado = pasta_bronze / "dataset_seguros_padronizado.csv"
arquivo_normalizado = pasta_bronze / "dataset_seguros_normalizado.csv"

df_padronizado_completo.to_csv(arquivo_padronizado, index=False)
df_normalizado_completo.to_csv(arquivo_normalizado, index=False)

print(f"\nDataset padronizado salvo em: {arquivo_padronizado.resolve()}")
print(f"Dataset normalizado salvo em: {arquivo_normalizado.resolve()}")