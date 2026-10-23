# Compilação e treinamento - Dataset Moon

from tensorflow import keras

learning_rate = 1e-3
def compilar_modelo(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )
    # Exibir as configurações no terminal
    print("\nConfiguração da compilação:")
    print("Otimizador: Adam")
    print(f"Taxa de aprendizado: {learning_rate}")
    print("Função de perda: Binary Crossentropy")
    print("Métricas: Accuracy, Precision, Recall e AUC")

    return model