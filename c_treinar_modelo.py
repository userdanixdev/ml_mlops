import torch
import torch.nn as nn
from b_preparar_tensores import X,y 

class Model(nn.Module):
    def __init__(self,hidden_neurons=2):
        super().__init__()
        self.hidden_neurons = hidden_neurons
        self.input_layer = nn.Linear(in_features=1,out_features=self.hidden_neurons)
        self.activation = nn.Sigmoid()
        self.output_layer = nn.Linear(in_features=self.hidden_neurons,out_features=1)
        

    def forward(self,x):
        out = self.input_layer(x)
        out = self.activation(out)
        out = self.output_layer(out)
        return out

EPOCHS = 1000
LR = 0.01

model = Model(hidden_neurons=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=LR)

weights = []
bias = []
for epoch in range(EPOCHS):
    outputs = model(X)
    loss = criterion(outputs,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weights.append(model.input_layer.weight.detach().clone())
    bias.append(model.input_layer.bias.detach().clone())
    if (epoch + 1) % 100 == 0:
        print(f"Época [{epoch + 1}/{EPOCHS}] - Loss: {loss.item():.6f}")


print(f'\nCamada de entrada: ')
print(f'Pesos: ', model.input_layer.weight.detach().view(-1).numpy())
print(f'Bias: ', model.input_layer.bias.detach().numpy())

print(f'\nCamada Escondida: ')
print(f'Pesos:', model.output_layer.weight.detach().view(-1).numpy())
print(f'Bias:', model.output_layer.bias.detach().numpy())


