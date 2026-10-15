
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Caminhos
arquivo = Path("src/bronze/dataset_seguros_normalizado.csv")
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

# Carregar dataset normalizado
df = pd.read_csv(arquivo)

# Separar registros conhecidos e ausentes
df_conhecidos = df[df["seguro_normalizado"].notna()]
df_ausentes = df[df["seguro_normalizado"].isna()]

# Criar gráfico
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df_conhecidos["idade_normalizada"],
    df_conhecidos["seguro_normalizado"],
    color="royalblue",
    label="Seguro conhecido",
    alpha=0.8
)

# Marcar registros sem valor de seguro
ax.scatter(
    df_ausentes["idade_normalizada"],
    [0] * len(df_ausentes),
    color="red",
    marker="x",
    s=70,
    label="Seguro ausente (marcado em 0)"
)

ax.set_title("Vértice Seguros — Dados normalizados")
ax.set_xlabel("Idade normalizada")
ax.set_ylabel("Valor do seguro normalizado")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

# Salvar e exibir
saida = pasta_saida / "dispersao_normalizada.png"
fig.savefig(saida, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {saida}")

plt.show()
