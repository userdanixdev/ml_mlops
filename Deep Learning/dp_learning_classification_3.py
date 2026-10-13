import pandas as pd
import torch

# Leitura do arquivo:

df = pd.read_excel(
    'alunos_base_original.xlsx'
)
print('Total de registros:', len(df))

# Definindo as features:

features = [
    'estudou',
    'conhecimento_previo',
    'facilidade_aprendizado'
]

# Separando os dados:

df_treino = df[
    df['resultado'].notna()
].copy()

# Esse é o dataset cujo os alunos estão sem resultado.
# E que o modelo deverá classificar posteriormente.
df_previsao = df[
    df['resultado'].isna()
].copy()

# Verificação da separação:

print(
    'Alunos para treinamento: ',
    len(df_treino)
)
print(
    'Alunos para previsão: ',
    len(df_previsao)
)

## Criando X para treinamento:

x_treino = torch.tensor(
    df_treino[features].values.copy(),
    dtype=torch.float32
)

## Criando Y do treinamento:

y_treino = torch.tensor(
    df_treino['resultado'].values.copy(),
    dtype=torch.float32
).unsqueeze(1)

## Criando o X dos alunos a prever: O ALVO

x_previsao = torch.tensor(
    df_previsao[features].values.copy(),
    dtype=torch.float32
)

## Obs: Não temos o y_previsao porque justamente não sabemos o resultado verdadeiro.

## Verificando os tensores:

print()
print('X treinamento:', x_treino.shape)
print('Y treinamento:', y_treino.shape)
print('X previsão:', x_previsao.shape)


# Vemos construir e treinar o modelo usando os 8.000 alunos que possuem resultado.
# Criando o modelo:

import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(
            in_features=3,
            out_features=1,
            bias=True
        )
    def forward(self,x):
        out = self.input_layer(x)
        return out

# Configurações do treinamento:
# 
EPOCHS = 1000
LR = 0.01
model = Model()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LR
) 

## Treinamento:

losses = []
for epoch in range(EPOCHS):
    # Forward:
    outputs = model(x_treino)
    # Calcula o erro:
    loss = criterion(
        outputs,
        y_treino
    )
    # Limpa os gradientes:
    optimizer.zero_grad()
    # Calcula os gradientes:
    loss.backward()
    # Atualiza os parâmetros:
    optimizer.step()
    # Guarda o erro:
    losses.append(
        loss.item()
    )

print()
print("Se o loss cair, significa que o modelo conseguiu reduzir o erro durante o treinamento.")
print('Loss inicial:', losses[0])
print('Loss final:', losses[-1])
print()
print(
    'Pesos:',
    model.input_layer.weight.data
)
print(
    'Bias:',
    model.input_layer.bias.data
)

with torch.no_grad():
    outputs_previsao = model(x_previsao)
    probabilidades = torch.sigmoid(outputs_previsao)
    previsoes = (probabilidades >= 0.5).float()

print()
print('Previsões realizadas:', len(previsoes))
print()
print('Primeiras probabilidades:')
print(probabilidades[:10])

print()
print('Primeiras previsões:')
print(previsoes[:10])


