
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Ler o arquivo original
df = pd.read_excel('alunos_base_original.xlsx')

# 2. Separar os registros com resultado conhecido
df_treino_teste = df[df['resultado'].notna()].copy()

features = [
    'estudou',
    'conhecimento_previo',
    'facilidade_aprendizado'
]

X = df_treino_teste[features].values.copy()
y = df_treino_teste['resultado'].values.copy()

# 3. Separar treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Converter os dados para tensores PyTorch
X_treino = torch.tensor(X_treino, dtype=torch.float32)
X_teste = torch.tensor(X_teste, dtype=torch.float32)

y_treino = torch.tensor(
    y_treino, dtype=torch.float32
).unsqueeze(1)

y_teste = torch.tensor(
    y_teste, dtype=torch.float32
).unsqueeze(1)

# 5. Definir a rede neural
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(
            in_features=3,
            out_features=1,
            bias=True
        )

    def forward(self, x):
        return self.input_layer(x)

# 6. Instanciar o modelo e configurar o treinamento
torch.manual_seed(42)

model = Model()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

EPOCHS = 1000

# 7. Treinar somente com X_treino e y_treino
for epoch in range(EPOCHS):
    model.train()

    outputs = model(X_treino)
    loss = criterion(outputs, y_treino)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 8. Fazer previsões nos dados de teste
model.eval()

with torch.no_grad():
    logits = model(X_teste)
    probabilidades = torch.sigmoid(logits)
    previsoes = (probabilidades >= 0.5).float()

y_real = y_teste.numpy().ravel().astype(int)
y_pred = previsoes.numpy().ravel().astype(int)

# 9. Calcular a matriz de confusão e as métricas
matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])

accuracy = accuracy_score(y_real, y_pred)
precision = precision_score(y_real, y_pred, zero_division=0)
recall = recall_score(y_real, y_pred, zero_division=0)
f1 = f1_score(y_real, y_pred, zero_division=0)

# 10. Exibir os resultados
print('=' * 50)
print('AVALIAÇÃO DO MODELO')
print('=' * 50)

print(f'Total de registros conhecidos: {len(X)}')
print(f'Registros de treinamento: {len(X_treino)}')
print(f'Registros de teste: {len(X_teste)}')
print(f'Loss final: {loss.item():.4f}')

print('\nMatriz de confusão:')
print(matriz)

print('\nMétricas:')
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1-score:  {f1:.4f}')

print('\nRelatório de classificação:')
print(
    classification_report(
        y_real,
        y_pred,
        labels=[0, 1],
        target_names=['Reprovado (0)', 'Aprovado (1)'],
        zero_division=0
    )
)