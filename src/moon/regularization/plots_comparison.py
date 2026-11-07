
# Gráfico comparativo dos estudos
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plotar_comparativo(resultados_experimentos):

    # Resultados do primeiro estudo, sem regularização
    resultados_originais = {
        "accuracy": 0.9400,
        "auc": 0.9901,
        "loss": 0.1672,
        "precision": 0.9490,
        "recall": 0.9300
    }

    # Organizar os resultados
    nomes = ["Original"] + list(resultados_experimentos.keys())

    resultados = [resultados_originais]

    for experimento in resultados_experimentos.values():
        resultados.append(experimento["resultados"])

    # Métricas de desempenho
    metricas = ["accuracy", "auc", "precision", "recall"]

    valores = {
        metrica: [
            resultado[metrica]
            for resultado in resultados
        ]
        for metrica in metricas
    }

    # Criar figura
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    x = np.arange(len(nomes))
    largura = 0.18

    # Gráfico das métricas
    for i, metrica in enumerate(metricas):

        deslocamento = (i - 1.5) * largura

        barras = ax1.bar(
            x + deslocamento,
            valores[metrica],
            largura,
            label=metrica.capitalize()
        )

        ax1.bar_label(
            barras,
            fmt="%.3f",
            padding=3,
            fontsize=8
        )

    ax1.set_title("Métricas de desempenho")
    ax1.set_ylabel("Resultado")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes)
    ax1.set_ylim(0.88, 1.04)
    ax1.legend(ncols=4)
    ax1.grid(axis="y", alpha=0.3)

    # Gráfico da função de perda
    valores_loss = [
        resultado["loss"]
        for resultado in resultados
    ]

    barras_loss = ax2.bar(
        x,
        valores_loss,
        width=0.5,
        label="Loss"
    )

    ax2.bar_label(
        barras_loss,
        fmt="%.4f",
        padding=3
    )

    ax2.set_title("Função de perda")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nomes)
    ax2.grid(axis="y", alpha=0.3)

    # Configurações finais
    fig.suptitle(
        "Comparação dos estudos - Dataset Moon",
        fontsize=14
    )

    plt.tight_layout()

    # Salvar gráfico
    root = Path(__file__).resolve().parents[3]
    caminho = root / "docs" / "images" / "comparativo_regularizacao.png"

    caminho.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(caminho, dpi=300, bbox_inches="tight")

    print(f"\nGráfico salvo em: {caminho}")

    plt.show()


# Execução independente para testar o gráfico
#if __name__ == "__main__":

    resultados_experimentos = {
        "Dropout": {
            "resultados": {
                "accuracy": 0.9900,
                "auc": 0.9997,
                "loss": 0.0441,
                "precision": 0.9804,
                "recall": 1.0000
            }
        },
        "BatchNormalization": {
            "resultados": {
                "accuracy": 0.9900,
                "auc": 0.9998,
                "loss": 0.0363,
                "precision": 0.9804,
                "recall": 1.0000
            }
        },
        "L2": {
            "resultados": {
                "accuracy": 0.9900,
                "auc": 1.0000,
                "loss": 0.0438,
                "precision": 0.9804,
                "recall": 1.0000
            }
        },
        "Dropout + L2": {
            "resultados": {
                "accuracy": 0.9900,
                "auc": 0.9996,
                "loss": 0.0560,
                "precision": 0.9804,
                "recall": 1.0000
            }
        }
    }

    plotar_comparativo(resultados_experimentos)
