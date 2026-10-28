
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Fronteira de decisão com dados originais - Dataset Moon
def plotar_fronteira_decisao_2(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    scaler
):
    # Caminho da raiz do projeto
    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    # Converter os dados para arrays NumPy
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Recuperar os valores originais, sem normalização
    X_train_original = scaler.inverse_transform(X_train)
    X_test_original = scaler.inverse_transform(X_test)

    X_todos = np.vstack([
        X_train_original,
        X_test_original
    ])

    y_todos = np.concatenate([
        y_train,
        y_test
    ])

    # Identificar erros de classificação no teste
    erros = y_test != y_pred

    # Criar grade usando as escalas originais
    margem = 0.15

    x1 = np.linspace(
        X_todos[:, 0].min() - margem,
        X_todos[:, 0].max() + margem,
        400
    )

    x2 = np.linspace(
        X_todos[:, 1].min() - margem,
        X_todos[:, 1].max() + margem,
        400
    )

    xx1, xx2 = np.meshgrid(x1, x2)

    grade_original = np.c_[
        xx1.ravel(),
        xx2.ravel()
    ]

    # Normalizar a grade para o modelo fazer as previsões
    grade_normalizada = scaler.transform(grade_original)

    probabilidades = model.predict(
        grade_normalizada,
        verbose=0
    ).reshape(xx1.shape)

    # Estilo visual
    sns.set_theme(style="white", context="notebook")

    fig, ax = plt.subplots(figsize=(11, 8))

    # Regiões previstas pelo modelo
    ax.contourf(
        xx1,
        xx2,
        probabilidades,
        levels=[0, 0.5, 1],
        colors=["#B9DDF2", "#FFD1A8"],
        alpha=0.75
    )

    # Fronteira de decisão
    ax.contour(
        xx1,
        xx2,
        probabilidades,
        levels=[0.5],
        colors="#202020",
        linewidths=2.5
    )

    # Todos os dados: valores originais e classes verdadeiras
    sns.scatterplot(
        x=X_todos[:, 0],
        y=X_todos[:, 1],
        hue=y_todos,
        hue_order=[0, 1],
        palette={
            0: "#075985",
            1: "#C2410C"
        },
        s=48,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.6,
        legend="full",
        ax=ax
    )

    # Destacar erros encontrados no conjunto de teste
    if np.any(erros):
        ax.scatter(
            X_test_original[erros, 0],
            X_test_original[erros, 1],
            marker="X",
            s=150,
            c="#D00000",
            edgecolors="white",
            linewidths=0.8,
            label="Erros no teste",
            zorder=5
        )

    # Configurar o gráfico
    ax.set_title(
        "Fronteira de Decisão - Dataset Moon",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("x1 ", fontsize=12)
    ax.set_ylabel("x2 ", fontsize=12)

    ax.legend(
        title="Classe real",
        loc="best",
        frameon=True
    )

    sns.despine()
    fig.tight_layout()

    # Salvar o gráfico
    fig.savefig(
        pasta_imagens / "fronteira_decisao_completa.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
