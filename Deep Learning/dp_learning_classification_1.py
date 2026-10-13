import torch
import torch.nn as nn
import pandas as pd

dados = {
    'nome': [
        'Ana',
        'Bruno',
        'Carlos',
        'Daniel',
        'Eduardo',
        'Fernanda',
        'Gabriel',
        'Helena',
        'Igor',
        'Juliana'
    ],
    'estudou':[
        0.9,0.3,0.8,0.4,0.7,
        0.2,0.95,0.5,0.6,0.1
    ],
    'conhecimento_previo':[
        0.8,0.2,0.7,0.3,0.6,
        0.1,0.9,0.4,0.5,0.2
    ],
    'facilidade_aprendizado':[
        0.9,0.4,0.8,0.3,0.7,
        0.2,0.95,0.5,0.6,0.1
    ],
    'resultado':[
        1,0,1,0,1,
        0,1,0,1,0
    ]
}
df = pd.DataFrame(dados)
print(df)

# Temos aqui um dataset supervisionado em que:

# X = características do estudante: estudou, conhecimento previo e facilidade de aprendizado.
# Y = Variável dependente em que temos o resultado se passou ou não.
# A variável 'nome' não influencia em nada na decisão de previsão.
# Portanto:

features = [
    'estudou',
    'conhecimento_previo',
    'facilidade_aprendizado'
]

# Transformando o dataframe em tensores ( arrays )

x = torch.FloatTensor(
    df[['estudou','conhecimento_previo','facilidade_aprendizado']].values
)

y = torch.FloatTensor(df['resultado'].values)
y= y.unsqueeze(1)

print(x)
print(x.shape)
print(y)

# O resultado do print temos: [10,3], ou seja, 10 amostras com 3 características.

# Criando a rede neural:

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

EPOCHS = 1000
LR = 0.01
model = Model()
criterion = nn.BCEWithLogitsLoss()
# BCE -> Binary Cross Entropy -> Específica para problemas de classificação binária.
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr = LR
)    

losses = []
for epoch in range(EPOCHS):
    outputs = model(x)
    loss = criterion(outputs,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f'Loss Inicial: {losses[0]:.4f}')
print(f'\nLoss Final: {losses[-1]:.4f}')

# A previsão: O resultado sairá um 'logit', gerado pelo PyTorch, então:

outputs = model(x) # Os outputs não são as probabilidades e sim resultados matemáticos gerados
# Conhecidos como 'logit'

# Transformando 'logits' em probabilidades:

probabilidade = torch.sigmoid(outputs)
print(probabilidade)
previsao = (probabilidade >= 0.5).float()
# Agora temos o y que são os primeiros resultados e 'previsão' que é o resultado previsto.

print('Real:')
print(y)

print('Previsão:')
print(previsao)

# Calculando a acurácia:

acertos = (previsao == y).sum()
total = y.size(0)
acuracia = acertos / total
print(f'\nAcurácia: {acuracia.item()*100:.2f}%')

# Criando uma tabela de resultados

resultado = pd.DataFrame({
    'nome':df['nome'],
    'resultado_real':y.squeeze().numpy(),
    'probabilidade':probabilidade.squeeze().detach().numpy(),
    'previsao':previsao.squeeze().numpy()
})

print('\n Resultado das previsões: ')
print(resultado)

# Matriz de confusão:

tp = ((previsao == 1) & (y == 1)).sum()
tn = ((previsao == 0) & (y == 0)).sum()
fp = ((previsao == 1) & (y == 0)).sum()
fn = ((previsao == 0) & (y == 1)).sum()

print('\nMatriz de confusão:')
print(f'TP (Verdadeiro Positivo): {tp.item()}')
print(f'TN (Verdadeiro Negativo): {tn.item()}')
print(f'FP (Falso Positivo): {fp.item()}')
print(f'FN (Falso Negativo): {fn.item()}')

# Calculo da precisão:

precisao = tp / (tp + fp)
print(f'Precisão: {precisao.item() * 100:.2f}%')

# Recall:
# Dos alunos que realmente passaram, quantos o modelo conseguiu identificar?
recall = tp/(tp+fn)
print(f'Recall: {recall.item()*100:.2f}%')

# F1-Score: Procura equilibrar a precisão + Recall

f1 = 2*(precisao*recall)/(precisao+recall)
print(f'F1-Score:{f1.item()*100:.2f}%')

# Acurácia  → acertos gerais
# Precisão  → qualidade das previsões positivas
# Recall    → capacidade de encontrar os positivos
# F1        → equilíbrio entre precisão e recall

# Só um cuidado: como o dataset tem apenas 10 alunos e foi construído de forma bastante simples,
# essas métricas podem ficar em 100%. Isso não significa que o modelo seria excelente em dados reais;
# significa apenas que ele conseguiu separar muito bem esses 10 exemplos.
# Para avaliar um modelo de verdade, o próximo passo seria separar os dados em treino e teste e
#  calcular as métricas no conjunto que o modelo nunca viu.