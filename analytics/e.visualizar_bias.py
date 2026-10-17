
import numpy as np
import matplotlib.pyplot as plt
from c_treinar_modelo import Model, model
from d_visualizar_ativacoes import saida_1, saida_2, bias_saida, x_numpy, tensor_to_numpy, pasta_saida
import torch 
from pathlib import Path

# Caminhos
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

# Soma das contribuições dos dois perceptrons: com e sem bias
# Soma das contribuições sem o bias de saída
soma_sem_bias = saida_1 + saida_2

# Soma das contribuições com o bias de saída
soma_com_bias = soma_sem_bias + bias_saida[0]

# Conversão para NumPy
soma_sem_bias_numpy = tensor_to_numpy(soma_sem_bias).ravel()
soma_com_bias_numpy = tensor_to_numpy(soma_com_bias).ravel()

# Criar gráfico
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    x_numpy,
    soma_sem_bias_numpy,
    label="Soma dos perceptrons sem bias"
)

ax.plot(
    x_numpy,
    soma_com_bias_numpy,
    label="Soma dos perceptrons com bias"
)

ax.set_xlabel("Idade normalizada")
ax.set_ylabel("Valor previsto normalizado")
ax.set_title("Soma dos perceptrons com e sem bias")
ax.legend()
ax.grid(True)

# Salvar o gráfico
saida = pasta_saida / "soma_perceptrons_com_sem_bias.png"
fig.savefig(saida, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {saida}")

plt.show()
plt.close(fig)