# Regularização
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.moon.regularization.plots_comparison import plotar_comparativo
from src.moon.regularization.plots_threshold import (
    plotar_fronteiras_limiares
)

def carregar_dados():
    root = Path(__file__).resolve().parents[3]
    path_csv = root / "data" / "dataset_moon.csv"
    df = pd.read_csv(path_csv)
    return df

def preparar_dados(df):
    # Separação das características e do alvo
    X = df.drop(columns=["classe"])
    y = df["classe"]

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Total de registros: {len(df)}")
    print(f"X_train: {len(X_train)}")
    print(f"y_train: {len(y_train)}")
    print(f"X_test: {len(X_test)}")
    print(f"y_test: {len(y_test)}")

    # Padronização das características
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def criar_modelo(
        n_caracteristicas,
        neuronios=(32,16),
        regularizacao=None,
        dropout_rate=0.2,
        l2_rate=0.0001
        ):
    # Camada de entrada
    inputs = keras.Input(shape=(n_caracteristicas,))
    x = inputs
    # Percorrer as camadas ocultas
    for unidades in neuronios:
        # Regularização dos pesos com L2
        regularizador = (
            keras.regularizers.l2(l2_rate)
            if regularizacao in ["l2","dropout_l2"]
            else None
        )
        x = layers.Dense(
            unidades,
            activation="relu",
            kernel_regularizer=regularizador
        )(x)
        # BatchNormalization
        if regularizacao == "batch_normalization":
            x = layers.BatchNormalization()(x)
        # Dropout
        if regularizacao in ["dropout","dropout_l2"]:
            x = layers.Dropout(dropout_rate)(x)

    # Camada de saída para classificação binária
    outputs = layers.Dense(1,activation="sigmoid")(x)
    # Construção do modelo
    model = keras.Model(inputs=inputs, outputs=outputs)       

    return model

# O parâmetro regularizacao aceita : "dropout", "batch_normalization", "l2" e None, para não aplicar essas técnicas.
# essa alteração existe apenas em regulate.py. O b_model.py original continua com as duas camadas de 16 neurônios.

def executar_experimentos(X_train, X_test, y_train, y_test):

    configuracoes = {
        "Dropout": "dropout",
        "BatchNormalization": "batch_normalization",
        "L2": "l2",
        "Dropout + L2": "dropout_l2"
    }

    resultados_experimentos = {}

    for nome, tecnica in configuracoes.items():
        keras.utils.set_random_seed(42)
        print(f"\n===== EXPERIMENTO: {nome} =====")

        model = criar_modelo(
            n_caracteristicas=X_train.shape[1],
            neuronios=(32, 16),
            regularizacao=tecnica
        )

        model = compilar_modelo(model, learning_rate=0.001)

        history = treinar_modelo(
            model,
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            patience=10
        )

        resultados, y_pred_prob = avaliar_modelo(
            model,
            X_test,
            y_test
        )

        resultados_experimentos[nome] = {
            "model": model,
            "history": history,
            "resultados": resultados,
            "y_pred_prob": y_pred_prob
        }

    return resultados_experimentos

def compilar_modelo(model, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model

def treinar_modelo(
        model,
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        patience=10
):
    # Monitoramento do treinamento
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Treinamento
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )

    return history

def avaliar_modelo(model, X_test, y_test):
    # Avaliação do modelo com os dados de teste
    resultados = model.evaluate(
        X_test,
        y_test,
        return_dict=True,
        verbose=0
    )

    # Previsão das probabilidades
    y_pred_prob = model.predict(
        X_test,
        verbose=0
    ).ravel()

    # Exibição dos resultados
    print("\n===== AVALIAÇÃO DO MODELO =====")

    for metrica, valor in resultados.items():
        print(f"{metrica}: {valor:.4f}")

    return resultados, y_pred_prob

def main():

    df = carregar_dados()

    X_train, X_test, y_train, y_test, scaler = preparar_dados(df)

    resultados = executar_experimentos(
        X_train,
        X_test,
        y_train,
        y_test
    )
      # Gerar gráfico comparativo dos estudos
    plotar_comparativo(resultados)
# Fronteiras de decisão com busca do melhor limiar
    plotar_fronteiras_limiares(
    resultados,
    X_test,
    y_test,
    scaler
)
    
if __name__ == "__main__":
    main()