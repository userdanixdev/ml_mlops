import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(
    "analytics/predicoes/previsoes_seguros.csv"
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["idade"],
    df["valor_seguro_previsto"]
)

ax.set_title("Idade x valor previsto do seguro")
ax.set_xlabel("Idade")
ax.set_ylabel("Valor do seguro previsto (R$)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "analytics/graficos/idade_vs_valor_previsto.png",
    dpi=300
)
plt.show()