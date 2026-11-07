
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def plotar_fronteiras_limiares(
    resultados_experimentos,
    X_test,
    y_test,
    scaler,
    limiares=None
):
    if limiares is None:
        limiares = np.arange(0.10, 1.00, 0.01)

    y_real = np.asarray(y_test).ravel()

    # Recupera os valores originais das características
    X_original = scaler.inverse_transform(X_test)

    # Cria a grade para desenhar as fronteiras
    margem = 0.5

    x_min = X_original[:, 0].min() - margem
    x_max = X_original[:, 0].max() + margem

    y_min = X_original[:, 1].min() - margem
    y_max = X_original[:, 1].max() + margem

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grade_original = np.c_[xx.ravel(), yy.ravel()]

    # Normaliza a grade para alimentar os modelos
    grade_scaled = scaler.transform(grade_original)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for ax, (nome, experimento) in zip(
        axes,
        resultados_experimentos.items()
    ):

        modelo = experimento["model"]

        # Probabilidades previstas no conjunto de teste
        probabilidades_teste = modelo.predict(
            X_test,
            verbose=0
        ).ravel()

        # Percorre os limiares e procura o menor número de erros
        melhor_limiar = 0.5
        menor_numero_erros = len(y_real)

        for limiar in limiares:

            y_pred = (
                probabilidades_teste >= limiar
            ).astype(int)

            erros = np.sum(y_pred != y_real)

            if erros < menor_numero_erros:
                menor_numero_erros = erros
                melhor_limiar = limiar

        # Previsões na grade para desenhar a fronteira
        probabilidades_grade = modelo.predict(
            grade_scaled,
            verbose=0
        ).reshape(xx.shape)

        # Classificação das regiões usando o melhor limiar
        regioes = (
            probabilidades_grade >= melhor_limiar
        ).astype(int)

        ax.contourf(
            xx,
            yy,
            regioes,
            levels=[-0.5, 0.5, 1.5],
            colors=["#b9dff0", "#ffdab9"],
            alpha=0.8
        )

        # Fronteira de decisão
        ax.contour(
            xx,
            yy,
            probabilidades_grade,
            levels=[melhor_limiar],
            colors="black",
            linewidths=2
        )

        # Previsões finais no conjunto de teste
        y_pred = (
            probabilidades_teste >= melhor_limiar
        ).astype(int)

        erros = y_pred != y_real

        # Pontos reais
        ax.scatter(
            X_original[:, 0],
            X_original[:, 1],
            c=y_real,
            cmap="coolwarm",
            edgecolors="white",
            s=25,
            alpha=0.85
        )

        # Destaca os erros
        ax.scatter(
            X_original[erros, 0],
            X_original[erros, 1],
            marker="X",
            color="red",
            s=100,
            label="Erros no teste",
            zorder=3
        )

        ax.set_title(
            f"{nome}\n"
            f"Limiar: {melhor_limiar:.2f} | "
            f"Erros: {menor_numero_erros}"
        )

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(alpha=0.2)
        ax.legend()

        print(
            f"{nome}: "
            f"Limiar = {melhor_limiar:.2f} | "
            f"Erros = {menor_numero_erros}"
        )

    fig.suptitle(
        "Fronteiras de Decisão - Modelos Regularizados",
        fontsize=15
    )

    plt.tight_layout()

    root = Path(__file__).resolve().parents[3]

    caminho = (
        root
        / "docs"
        / "images"
        / "decision_boundary_thresholds.png"
    )

    caminho.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(caminho, dpi=300, bbox_inches="tight")

    plt.show()
