
import matplotlib.pyplot as plt
from dataset_moon import df

# Gráfico de dispersão:
# Identificar as características e a classe
colunas_caracteristicas = df.columns.drop("classe")

# Criar o gráfico
plt.figure(figsize=(10, 6))

for classe in sorted(df["classe"].unique()):
    dados_classe = df[df["classe"] == classe]

    plt.scatter(
        dados_classe[colunas_caracteristicas[0]],
        dados_classe[colunas_caracteristicas[1]],
        label=f"Classe {classe}",
        alpha=0.7
    )

plt.title("Dataset Moon — Distribuição das Classes")
plt.xlabel(colunas_caracteristicas[0])
plt.ylabel(colunas_caracteristicas[1])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Salvar e exibir o gráfico
plt.savefig("docs/images/scatter_dataset_moon.png", dpi=300, bbox_inches="tight")
plt.show()
