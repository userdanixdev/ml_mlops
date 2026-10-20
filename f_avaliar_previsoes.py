
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from b_preparar_tensores import X, df_treino
from c_treinar_modelo import model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np


# Colocar o modelo em modo de avaliação
model.eval()

# Gerar previsões para as 85 idades conhecidas
with torch.no_grad():
    previsoes_normalizadas = model(X).cpu().numpy().ravel()

# Criar tabela para comparar os valores
df_avaliacao = df_treino[["nome", "idade", "valor_seguro"]].copy()

df_avaliacao["seguro_normalizado_previsto"] = previsoes_normalizadas

print("\n===== COMPARAÇÃO DOS 85 REGISTROS =====")
print(df_avaliacao.to_string(index=False))
print(f"\nTotal de registros avaliados: {len(df_avaliacao)}")

# Converter as previsões normalizadas para valores reais (R$)
df_original = pd.read_csv("src/dataset_seguros_vertice.csv")

seguros_conhecidos = (
    df_original.loc[
        df_original["valor_seguro"].notna(),
        ["valor_seguro"]
    ].to_numpy()
)

scaler_y = MinMaxScaler()
scaler_y.fit(seguros_conhecidos)

previsoes_reais = scaler_y.inverse_transform(
    previsoes_normalizadas.reshape(-1, 1)
).ravel()

# Adicionar as previsões em reais à tabela existente
df_avaliacao["valor_seguro_previsto"] = previsoes_reais

print("\n===== PREVISÕES CONVERTIDAS PARA REAIS =====")
print(df_avaliacao.to_string(index=False))

# Valores reais e previsões em reais:
valores_reais = df_avaliacao["valor_seguro"].to_numpy()
valores_previstos = df_avaliacao["valor_seguro_previsto"].to_numpy()

# Calcular métricas:
mse = mean_squared_error(valores_reais, valores_previstos)
rmse = np.sqrt(mse)
mae = mean_absolute_error(valores_reais, valores_previstos)
r2 = r2_score(valores_reais, valores_previstos)

# Exibir resultados
print("\n===== MÉTRICAS DO MODELO EM REAIS =====")
print(f"MSE:  R$² {mse:,.2f}")
print(f"RMSE: R$   {rmse:,.2f}")
print(f"MAE:  R$   {mae:,.2f}")
print(f"R²:        {r2:.6f}")


# Calcular o erro absoluto de cada registro
df_avaliacao["erro_absoluto"] = (
    df_avaliacao["valor_seguro"]
    - df_avaliacao["valor_seguro_previsto"]
).abs()

# Salvar os resultados da avaliação
pasta_saida = Path("analytics/avaliacao_predicao")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivo_saida = pasta_saida / "avaliacao_predicao_seguros.csv"

df_avaliacao.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nAvaliação salva em: {arquivo_saida.resolve()}")

