
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Matriz de confusão - Dataset Moon
def plotar_matriz_confusao(y_test, y_pred):

    # Caminho da raiz do projeto
    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    # Converter os dados para arrays NumPy
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calcular a matriz de confusão
    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    # Criar o gráfico
    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[0, 1]
    )

    exibicao.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Matriz de Confusão - Dataset Moon")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe verdadeira")
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(
        pasta_imagens / "matriz_confusao.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
