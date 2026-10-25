
from pathlib import Path
import matplotlib.pyplot as plt


# Gráficos do treinamento - Dataset Moon
def plotar_historico(history):

    # Caminho da raiz do projeto
    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True, exist_ok=True)

    # Número das épocas
    epocas = range(1, len(history.history["loss"]) + 1)

    # Gráfico 1: Loss e Val Loss
    plt.figure(figsize=(9, 5))

    plt.plot(
        epocas,
        history.history["loss"],
        marker="o",
        label="Loss - Treino"
    )

    plt.plot(
        epocas,
        history.history["val_loss"],
        marker="o",
        label="Val Loss - Validação"
    )

    plt.title("Evolução da Perda durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Binary Crossentropy")
    plt.xticks(list(epocas))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        pasta_imagens / "loss_treino_validacao.png",
        dpi=150
    )
    plt.show()
    plt.close()

    # Gráfico 2: Accuracy e Val Accuracy
    plt.figure(figsize=(9, 5))

    plt.plot(
        epocas,
        history.history["accuracy"],
        marker="o",
        label="Accuracy - Treino"
    )

    plt.plot(
        epocas,
        history.history["val_accuracy"],
        marker="o",
        label="Val Accuracy - Validação"
    )

    plt.title("Evolução da Acurácia durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.xticks(list(epocas))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        pasta_imagens / "accuracy_treino_validacao.png",
        dpi=150
    )
    plt.show()
    plt.close()
