
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Fronteira de decisão - Dataset Moon
def plotar_fronteira_decisao(model, X_test, y_test):

    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).ravel()

    # Criar a grade de coordenadas
    margem = 0.5

    x1 = np.linspace(X_test[:, 0].min() - margem,
                     X_test[:, 0].max() + margem, 200)

    x2 = np.linspace(X_test[:, 1].min() - margem,
                     X_test[:, 1].max() + margem, 200)

    xx1, xx2 = np.meshgrid(x1, x2)
    grade = np.c_[xx1.ravel(), xx2.ravel()]

    # Prever as probabilidades
    probabilidades = model.predict(grade, verbose=0)
    probabilidades = probabilidades.reshape(xx1.shape)

    # Plotar as regiões e a fronteira
    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx1, xx2, probabilidades,
        levels=[0, 0.5, 1],
        alpha=0.25,
        cmap="coolwarm"
    )

    plt.contour(
        xx1, xx2, probabilidades,
        levels=[0.5],
        colors="black",
        linewidths=1.5
    )

    # Plotar os pontos reais
    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        s=30
    )

    plt.title("Fronteira de Decisão - Dataset Moon")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()

    plt.savefig(
        pasta_imagens / "fronteira_decisao.png",
        dpi=150
    )

    plt.show()
    plt.close()
