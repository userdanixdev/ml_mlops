
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from dp_learning_regression_1 import X, y, X_prever
from treinar_modelo_2 import model

# Diretório de saída
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

# Dados originais para recuperar a escala monetária
df_original = pd.read_csv("src/dataset_seguros_vertice.csv")
valores_conhecidos = (
    df_original["valor_seguro"]
    .dropna()
    .to_numpy()
    .reshape(-1, 1)
)

scaler_y = MinMaxScaler()
scaler_y.fit(valores_conhecidos)

# Avaliar o modelo
model.eval()

with torch.no_grad():
    # Previsões nos dados conhecidos
    previsoes_treino_norm = model(X).cpu().numpy()

    # Previsões nos registros sem valor de seguro
    previsoes_ausentes_norm = model(X_prever).cpu().numpy()

    # Faixa contínua de idades normalizadas
    x_curva = torch.linspace(
        0, 1, 300
    ).reshape(-1, 1)

    # Saída completa da rede neural
    y_curva_norm = model(x_curva).cpu().numpy()

# Converter previsões para a escala original
valores_reais = scaler_y.inverse_transform(
    y.cpu().numpy()
).ravel()

valores_previstos = scaler_y.inverse_transform(
    previsoes_treino_norm
).ravel()

valores_ausentes = scaler_y.inverse_transform(
    previsoes_ausentes_norm
).ravel()

curva_seguro = scaler_y.inverse_transform(
    y_curva_norm
).ravel()

# Recuperar as idades originais
df_conhecidos = df_original.dropna(
    subset=["valor_seguro"]
).copy()

idade_min = df_original["idade"].min()
idade_max = df_original["idade"].max()

def idade_original(idade_normalizada):
    return idade_min + idade_normalizada * (
        idade_max - idade_min
    )

idades_curva = idade_original(
    x_curva.numpy().ravel()
)

idades_conhecidas = df_conhecidos["idade"].to_numpy()
idades_ausentes = X_prever.numpy().ravel()
idades_ausentes = idade_original(idades_ausentes)

# Gráfico: dados, previsões e curva aprendida
fig, ax = plt.subplots(figsize=(11, 7))

ax.scatter(
    idades_conhecidas,
    valores_reais,
    label="Valores reais conhecidos",
    alpha=0.75
)

ax.scatter(
    idades_conhecidas,
    valores_previstos,
    label="Previsões do modelo (treinamento)",
    marker="x"
)

ax.scatter(
    idades_ausentes,
    valores_ausentes,
    label="Previsões dos valores ausentes",
    marker="D"
)

ax.plot(
    idades_curva,
    curva_seguro,
    label="Curva aprendida pela rede neural",
    linewidth=2.5
)

ax.set_title(
    "Regressão com PyTorch: curva sigmoidal aprendida"
)
ax.set_xlabel("Idade")
ax.set_ylabel("Valor do seguro (R$)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    pasta_saida / "dispersao_curva_modelo.png",
    dpi=300
)
plt.show()
plt.close(fig)

print(
    "Gráfico salvo em:",
    (pasta_saida / "dispersao_curva_modelo.png").resolve()
)