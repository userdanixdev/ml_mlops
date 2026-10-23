# Arquitetura MLP - Dataset Moon
from tensorflow import keras
from tensorflow.keras import layers

def criar_modelo(n_caracteristicas):
    # Camada de entrada
    inputs = keras.Input(shape=(n_caracteristicas,))

    # Primeira camada oculta
    x = layers.Dense(16, activation="relu")(inputs)

    # Segunda camada oculta
    x = layers.Dense(16, activation="relu")(x)

    # Camada de saída para classificação binária
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Construção do modelo
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
