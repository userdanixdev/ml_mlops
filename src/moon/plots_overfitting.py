
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Gráfico de identificação visual do overfitting - Dataset Moon:
def plotar_overfitting(history):
    # Caminho da raiz do projeto
    root = Path(__file__).resolve().parents[2]
    pasta_imagens = root / "docs" / "images"
    pasta_imagens.mkdir(parents=True,exist_ok=True)
    # Número das épocas:
    epocas = range(1,len(history.history["loss"])+1)
    # Obter os valores reais:
    loss_treino = history.history["loss"]
    loss_validacao = history.history["val_loss"]
    # Identificação da época com menor perda de validação:
    melhor_epoca = loss_validacao.index(min(loss_validacao))
    menor_val_loss = loss_validacao[melhor_epoca]
    # Converter o índice para o número da época>
    melhor_epoca += 1
    # Gráfico de Loss e Val Loss:
    plt.figure(figsize=(9,5))
    plt.plot(
        epocas,
        loss_treino,
        marker = "o",
        label="Loss - Treino"
    )
    plt.plot(
            epocas,
            loss_validacao,
            marker = "o",
            label="Val Loss - Validação"
        )
    # Marcar a época de menor perda de validação
    plt.axvline(
        x=melhor_epoca,
        linestyle="--",
        linewidth=2,
        label=f"Menor Val Loss - Época {melhor_epoca}"
    )
     # Destacar o menor valor da perda de validação
    plt.annotate(
        f"Menor Val Loss: {menor_val_loss:.4f}",
        xy=(melhor_epoca, menor_val_loss),
        xytext=(10, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        fontsize=9
    )

    plt.title("Análise de Overfitting - Dataset Moon")
    plt.xlabel("Época")
    plt.ylabel("Binary Crossentropy")
    plt.xticks(list(epocas))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig(
        pasta_imagens / "overfitting_dataset_moon.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
