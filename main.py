from src.moon.a_data import carregar_dados, preparar_dados
from src.moon.b_model import criar_modelo
from src.moon.c_compile import compilar_modelo
from src.moon.d_train import treinar_modelo
from src.moon.e_evaluate import avaliar_modelo
from src.moon.plots import plotar_historico

# 1. Carregar os dados
df = carregar_dados()

# 2. Preparar os dados
X_train, X_test, y_train, y_test, scaler = preparar_dados(df)

# 3. Criar a arquitetura
model = criar_modelo(n_caracteristicas=X_train.shape[1])
model.summary()

# 4. Compilar
model = compilar_modelo(model)

# 5. Treinar
history = treinar_modelo(model, X_train, y_train)

# Gerar os gráficos do treinamento
plotar_historico(history)

# 6. Avaliar com os dados de teste
#resultado = avaliar_modelo(model, X_test, y_test)