
import numpy as np
import matplotlib.pyplot as plt
from c_treinar_modelo import Model, model
import torch 
from pathlib import Path

# Caminhos
pasta_saida = Path("analytics/graficos")
pasta_saida.mkdir(parents=True, exist_ok=True)

def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


# Valores de entrada para representar as funções
x_range = torch.arange(0, 1.0, 0.02).reshape(-1, 1)

# Parâmetros aprendidos da rede
pesos_entrada = model.input_layer.weight.detach()
bias_entrada = model.input_layer.bias.detach()

pesos_saida = model.output_layer.weight.detach()
bias_saida = model.output_layer.bias.detach()

# Primeiro Perceptron oculto
y1 = x_range * pesos_entrada[0, 0] + bias_entrada[0]
sigmoid_1 = torch.sigmoid(y1)
saida_1 = sigmoid_1 * pesos_saida[0, 0]

# Segundo Perceptron oculto
y2 = x_range * pesos_entrada[1, 0] + bias_entrada[1]
sigmoid_2 = torch.sigmoid(y2)
saida_2 = sigmoid_2 * pesos_saida[0, 1]

# Conversão para NumPy
x_numpy = tensor_to_numpy(x_range).ravel()
y1_numpy = tensor_to_numpy(y1).ravel()
sigmoid_1_numpy = tensor_to_numpy(sigmoid_1).ravel()
saida_1_numpy = tensor_to_numpy(saida_1).ravel()

y2_numpy = tensor_to_numpy(y2).ravel()
sigmoid_2_numpy = tensor_to_numpy(sigmoid_2).ravel()
saida_2_numpy = tensor_to_numpy(saida_2).ravel()

# Saída completa da rede, incluindo o bias de saída
with torch.no_grad():
    previsoes = model(x_range)

previsoes_numpy = tensor_to_numpy(previsoes).ravel()

# Visualização das ativações dos Perceptrons
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(x_numpy, sigmoid_1_numpy, label="Perceptron 1 - Sigmoid")
ax.plot(x_numpy, sigmoid_2_numpy, label="Perceptron 2 - Sigmoid")
ax.set_xlabel("Idade normalizada")
ax.set_ylabel("Ativação")
ax.set_title("Funções de ativação dos Perceptrons ocultos")
ax.legend()
ax.grid(True)
#ax.show()

# Salvar e exibir
saida = pasta_saida / "ativacoes_perceptrons.png"
fig.savefig(saida, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {saida}")

plt.show()
plt.close(fig)


# Visualização das contribuições e da saída final
fig,ax = plt.subplots(figsize=(10,5))

ax.plot(x_numpy, saida_1_numpy, label="Contribuição do Perceptron 1")
ax.plot(x_numpy, saida_2_numpy, label="Contribuição do Perceptron 2")
ax.plot(x_numpy, previsoes_numpy, label="Saída completa da rede")

ax.set_xlabel("Idade normalizada")
ax.set_ylabel("Valor previsto normalizado")
ax.set_title("Contribuições dos Perceptrons para a saída")
ax.legend()
ax.grid(True)
#ax.show()

# Salvar o gráfico
saida = pasta_saida / "contribuicoes_perceptrons.png"
fig.savefig(saida, dpi=300, bbox_inches="tight")
print(f"Gráfico salvo em: {saida}")

plt.show()
plt.close(fig)