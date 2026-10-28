
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Distribuição das probabilidades previstas - Dataset Moon
def plotar_probabilidades(y_test, y_pred_prob):

    # Caminho da raiz do projeto
    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    # Converter os dados para arrays NumPy
    y_test = np.asarray(y_test).ravel()
    probabilidades = np.asarray(y_pred_prob).ravel()

    # Separar as amostras pela classe real
    classe_0 = y_test == 0
    classe_1 = y_test == 1

    # Criar o gráfico
    plt.figure(figsize=(8, 7))

    # Previsões das amostras da classe real 0
    plt.scatter(
        probabilidades[classe_0],
        np.where(classe_0)[0],
        label="Classe real 0",
        alpha=0.7
    )

    # Previsões das amostras da classe real 1
    plt.scatter(
        probabilidades[classe_1],
        np.where(classe_1)[0],
        label="Classe real 1",
        alpha=0.7
    )

    # Limiar de classificação
    plt.axvline(
        0.5,
        color="black",
        linestyle="--",
        label="Limiar = 0.5"
    )

    # Configurar os eixos
    plt.title("Probabilidades Previstas por Amostra - Dataset Moon")
    plt.xlabel("Probabilidade prevista para a classe 1")
    plt.ylabel("Amostras de teste")
    plt.xlim(-0.05, 1.05)

    plt.legend()
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(
        pasta_imagens / "distribuicao_probabilidades.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
