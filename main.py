from src.moon.a_data import carregar_dados, preparar_dados
from src.moon.b_model import criar_modelo
from src.moon.c_compile import compilar_modelo
from src.moon.d_train import treinar_modelo
from src.moon.e_evaluate import avaliar_modelo
from src.moon.plots_loss_accuracy import plotar_historico
from src.moon.plots_overfitting import plotar_overfitting
from src.moon.plots_decision_boundary import plotar_fronteira_decisao
from src.moon.plots_probabilities import plotar_probabilidades
from src.moon.plots_matriz_confusion import plotar_matriz_confusao
from src.moon.plots_decision_boundary_2 import plotar_fronteira_decisao_2

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

# Plotar gráficos de possíveis overfitting:
plotar_overfitting(history)

# Avaliar com os dados de teste
resultado, y_pred, y_pred_prob  = avaliar_modelo(model, X_test, y_test)

# 1. Fronteira de decisão
plotar_fronteira_decisao(model, X_test, y_test)

# 2. Distribuição das probabilidades
plotar_probabilidades(y_test, y_pred_prob)

# 3. Matriz de confusão
plotar_matriz_confusao(y_test, y_pred)

# 4. Plotar fronteira de decisão ( dados completos )
plotar_fronteira_decisao_2(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    scaler
)