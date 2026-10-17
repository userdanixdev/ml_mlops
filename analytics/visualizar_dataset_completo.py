
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from dp_learning_regression_1 import X
from treinar_modelo_2 import model

# Arquivos
arquivo_original = Path("src/dataset_seguros_vertice.csv")
arquivo_previsoes = Path("analytics/predicoes/previsoes_seguros.csv")
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

# Dataset original e previsões
df = pd.read_csv(arquivo_original)
df_previsoes = pd.read_csv(arquivo_previsoes)

# Recuperar a escala monetária original
valores_conhecidos = (
    df["valor_seguro"].dropna().to_numpy().reshape(-1, 1)
)
scaler_y = MinMaxScaler()
scaler_y.fit(valores_conhecidos)

# Inserir as previsões nos registros que estavam ausentes
df["tipo_valor"] = "Real"
df["valor_seguro_plot"] = df["valor_seguro"]

previsoes_por_nome = df_previsoes.set_index("nome")[
    "valor_seguro_previsto"
]

ausentes = df["valor_seguro"].isna()
df.loc[ausentes, "valor_seguro_plot"] = (
    df.loc[ausentes, "nome"].map(previsoes_por_nome)
)
df.loc[ausentes, "tipo_valor"] = "Previsto"

# Gerar a curva da rede neural ao longo das idades normalizadas
model.eval()
with torch.no_grad():
    x_curva = torch.linspace(0, 1, 300).reshape(-1, 1)
    y_curva_norm = model(x_curva).cpu().numpy()

curva_reais = scaler_y.inverse_transform(y_curva_norm).ravel()

idade_min = df["idade"].min()
idade_max = df["idade"].max()
idades_curva = (
    idade_min + x_curva.numpy().ravel() * (idade_max - idade_min)
)

# Gráfico completo
fig, ax = plt.subplots(figsize=(12, 7))

reais = df[df["tipo_valor"] == "Real"]
previstos = df[df["tipo_valor"] == "Previsto"]

ax.scatter(
    reais["idade"],
    reais["valor_seguro_plot"],
    label="Valores reais do dataset",
    alpha=0.8
)

ax.scatter(
    previstos["idade"],
    previstos["valor_seguro_plot"],
    label="Valores ausentes previstos",
    marker="D",
    s=65
)

ax.plot(
    idades_curva,
    curva_reais,
    label="Curva aprendida pela rede neural",
    linewidth=2.5
)

ax.set_title("Dataset completo: valores reais e previsões do seguro")
ax.set_xlabel("Idade")
ax.set_ylabel("Valor do seguro (R$)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    pasta_saida / "dataset_completo_reais_previstos.png",
    dpi=300
)
plt.show()
plt.close(fig)

print("Gráfico salvo em:", (
    pasta_saida / "dataset_completo_reais_previstos.png"
).resolve())

print("\nQuantidade de registros por tipo:")
print(df["tipo_valor"].value_counts())