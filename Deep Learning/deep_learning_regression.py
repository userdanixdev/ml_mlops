# Introdução ao Deep Learning: Redes Neurais Artificiais

import torch
import torch.nn as nn
import pandas as pd

# Funão de conversão de temperatura:
def convert_celcius_to_farenheit(celcius):
    return (1.8*celcius)+32

# Lista de graus em temperatura:
temp_celcius = [-10,20,100]
temp_farenheit = [convert_celcius_to_farenheit(temp) for temp in temp_celcius]

# DataFrame dos dados em Pandas:
df = pd.DataFrame ({'Celcius':temp_celcius,'Farenheit':temp_farenheit})
print(df)

# Modificação das variáveis: Transformar as variáveis em tensores ( vetores )

x = torch.FloatTensor(df.Celcius.values.astype(float))
y = torch.FloatTensor(df.Farenheit.values.astype(float))
# Aqui transformamos os vetores em matrizes 3 x 1, formato esperado para uma rede neural:
y = y.unsqueeze(1)

print(x)
print(x.shape)

print(y)
print(y.shape)

# A classe 'Model' irá herdar 'nn.Module' uma classe base do PyTorch para criação de redes neurais:
class Model(nn.Module):
    def __init__(self):
        super().__init__()  # <- Inicializa a estrutura
# Aqui a rede recebe 1 variável de entrada 'Celsius' e a rede produz 1 resultado ( Farenheit )      
        self.input_layer = nn.Linear(in_features=1, out_features=1,bias=True)
# Obs: O Bias começa com valores iniciais e, durante o treinamento, vai ajustando weight e bias.        
    # Como os dados irão passar pela rede?
    def forward(self,x):
# Assim, entregue o valor de x, passe pela camada linear e guarde o resultado em out.        
        out = self.input_layer(x)
        return out    
# O que são essas variáveis?
# EPOCHS ou CICLOS , representa a passagem completa pelo conjunto de dados durante o treinamento.
# Então, as amostras terão seu ciclo percorrido durante 1000 vezes. Significa repetir o treinamento
EPOCHS = 1000
# LEARNING RATE: Taxa de aprendizado. Determina o tamanho do passo ao ajustar os treinamentos.
LR = 0.2

model = Model()
# Função de perda ( LOSS FUNCTION ) Quanto a previsão da rede está distante do valor correto?
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr = LR)

# A ideia central do treinamento é reduzir o loss ao longo das epochs.

x = x.view(x.size(0),-1)
weights = []
bias = []
for epoch in range(EPOCHS):
    outputs = model.forward(x)
    loss = criterion(outputs,y)
    optimizer.zero_grad()
    # Apaga os gradientes calculados anteriormente. O Pytorch  acumula gradientes
    loss.backward()
    # O 'back'calcula os gradientes desse erro em relação aos parâmetros:
    optimizer.step()
    # efetivamente corrige
# Ciclo: Faça uma previsão → calcule o erro → limpe os gradientes antigos → calcule os novos gradientes → atualize os parâmetros."
# Repete 1000 e salva em uma lista com 'append'.
    weights.append(model.input_layer.weight.item())
    bias.append(model.input_layer.bias.item())
print(f'Peso: {model.input_layer.weight.item():.2f}\n'
      f'Bias: {model.input_layer.bias.item():.2f}')    

