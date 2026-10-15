
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

# Carregar dataset
df = pd.read_csv("src/dataset_seguros_vertice.csv")

# Criar cópia para preservar os dados originais
df_scale = df.copy()

# Separar registros com valor de seguro conhecido
df_conhecidos = df_scale[df_scale["valor_seguro"].notna()].copy()

# Selecionar a variável de entrada (idade)
X = df_conhecidos[["idade"]]

# Selecionar o alvo (valor do seguro)
y = df_conhecidos[["valor_seguro"]]

# Padronização dos valores de entrada e alvo:
# Padronização com StandardScaler
scaler_x = StandardScaler()
idades_padronizadas = scaler_x.fit_transform(X).round(3)

scaler_y = StandardScaler()
seguros_padronizados = scaler_y.fit_transform(y).round(3)

# Normalização MinMaxScaler (0 a 1)
minmax_x = MinMaxScaler()
idades_normalizadas = minmax_x.fit_transform(X).round(3)

minmax_y = MinMaxScaler()
seguros_normalizados = minmax_y.fit_transform(y).round(3)

# DataFrames para visualizar os resultados
df_padronizado = pd.DataFrame({
    "idade_padronizada":idades_padronizadas.ravel(),
    "seguro_padronizado":seguros_padronizados.ravel()},
    index=X.index)
df_normalizado = pd.DataFrame({
    "idade_normalizada":idades_normalizadas.ravel(),
    "seguro_normalizado":seguros_normalizados.ravel()},
    index=X.index)

# Preservar os valores ausentes:

df_padronizado_completo = df_scale[
    ["nome","idade","valor_seguro"]
].join(df_padronizado)

df_normalizado_completo = df_scale[
    ["nome","idade","valor_seguro"]
].join(df_normalizado)

# Visualização dos resultados:

print("\nDados originais:")
print(df_scale.head(10))

print("\nDados padronizados:")
print(df_padronizado_completo.head(10))

print("\nDados normalizados:")
print(df_normalizado_completo.head(10))

print("\nMínimos após normalização:")
print(df_normalizado.min())

print("\nMáximos após normalização:")
print(df_normalizado.max())

print("\nQuantidade de valores ausentes preservados:")
print(df_normalizado_completo.isna().sum())

# Diretório camada Bronze:
pasta_bronze = Path("src") / "bronze"
pasta_bronze.mkdir(parents=True, exist_ok = True)

# Salvar resultados na camada bronze:

arquivo_padronizado = (
    pasta_bronze / "dataset_seguros_padronizado.csv"
)
df_padronizado_completo.to_csv(
    arquivo_padronizado,
    index=False
)
# Salvar dataset normalizado:
arquivo_normalizado = (
    pasta_bronze / "dataset_seguros_normalizado.csv"
)
df_normalizado_completo.to_csv(
    arquivo_normalizado,
    index=False
)

print(f"\nDataset padronizado salvo em: {arquivo_padronizado.resolve()}")
print(f"Dataset normalizado salvo em: {arquivo_normalizado.resolve()}")