import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Carregar o dataset salvo na pasta src:
df= pd.read_csv("src/dataset_seguros_vertice.csv")

# Separação dos registros de valores conhecidos e ausentes:

df_conhecidos = df[df["valor_seguro"].notna()]
df_ausentes = df[df["valor_seguro"].isna()]

# Gráfico de dispersão:

fig,ax = plt.subplots(figsize=(10,6))

ax.scatter(
    df_conhecidos["idade"],
    df_conhecidos["valor_seguro"],
    color="blue",
    label="Valor Conhecido",
    alpha=0.7
)
# Representar os registros sem valor
ax.scatter(
    df_ausentes["idade"],
    [0]*len(df_ausentes),
    color="red",
    marker="x",
    label="Valor Ausente (NaN) "
)
ax.set_title("Vértice Seguros - Idade X Valor do Seguro")
ax.set_xlabel("Idade do Segurado")
ax.set_ylabel("Valor do Seguro (R$)")
ax.legend()
ax.grid(True,alpha=0.3)
fig.tight_layout()


pasta_graficos = Path("analytics/graficos")
pasta_graficos.mkdir(parents=True, exist_ok=True)
# Salvar em PNG:
caminho_grafico = pasta_graficos / "idade_valor_seguro.png"
fig.savefig(
    caminho_grafico,
    dpi=300,
    bbox_inches="tight"
    )
print(f"Gráfico salvo em: {caminho_grafico.resolve()}")
plt.show()
plt.close(fig)