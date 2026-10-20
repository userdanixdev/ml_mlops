
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Carregar os resultados da avaliação
df = pd.read_csv("analytics/avaliacao_predicao/avaliacao_predicao_seguros.csv")

# Criar o gráfico
plt.figure(figsize=(10, 6))

plt.scatter(
    df["valor_seguro"],
    df["valor_seguro_previsto"],
    alpha=0.8,
    label="Previsões do modelo"
)

# Linha de referência: previsão perfeita
limite_min = min(
    df["valor_seguro"].min(),
    df["valor_seguro_previsto"].min()
)

limite_max = max(
    df["valor_seguro"].max(),
    df["valor_seguro_previsto"].max()
)

plt.plot(
    [limite_min, limite_max],
    [limite_min, limite_max],
    linestyle="--",
    label="Previsão perfeita"
)

plt.xlabel("Valor real do seguro (R$)")
plt.ylabel("Valor previsto pelo modelo (R$)")
plt.title("Avaliação do modelo PyTorch: valores reais × previstos")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar o gráfico
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivo_saida = pasta_saida / "avaliacao_valores_reais_previstos.png"

plt.savefig(arquivo_saida, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico salvo em: {arquivo_saida.resolve()}")