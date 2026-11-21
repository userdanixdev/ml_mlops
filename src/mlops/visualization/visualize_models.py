
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics():
    metrics_path = Path("reports/model_metrics.csv")

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Arquivo de métricas não encontrado: {metrics_path}"
        )

    return pd.read_csv(metrics_path)


def visualize_models():
    metrics = load_metrics()

    comparison = metrics[
        [
            "model",
            "accuracy",
            "malignant_precision",
            "malignant_recall",
            "malignant_f1-score",
            "benign_precision",
            "benign_recall",
            "benign_f1-score"
        ]
    ].copy()

    comparison = comparison.rename(columns={
        "accuracy": "Accuracy",
        "malignant_precision": "Malignant Precision",
        "malignant_recall": "Malignant Recall",
        "malignant_f1-score": "Malignant F1-score",
        "benign_precision": "Benign Precision",
        "benign_recall": "Benign Recall",
        "benign_f1-score": "Benign F1-score"
    })

    comparison = comparison.set_index("model")

    ax = comparison.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.8
    )

    ax.set_title("Comparação de desempenho dos modelos")
    ax.set_xlabel("Modelos")
    ax.set_ylabel("Pontuação")
    ax.set_ylim(0.90, 1.00)
    ax.legend(title="Métricas")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.4f",
            padding=3,
            fontsize=9
        )

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = Path("docs/images/model_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"\nGráfico salvo em: {output_path}")


def visualize_class_metrics():
    metrics = load_metrics()

    class_metrics = metrics[
        [
            "model",
            "malignant_precision",
            "malignant_recall",
            "malignant_f1-score",
            "benign_precision",
            "benign_recall",
            "benign_f1-score"
        ]
    ].copy()

    class_metrics = class_metrics.set_index("model")

    for class_name in ["malignant", "benign"]:
        comparison = class_metrics[
            [
                f"{class_name}_precision",
                f"{class_name}_recall",
                f"{class_name}_f1-score"
            ]
        ].copy()

        comparison.columns = [
            "Precision",
            "Recall",
            "F1-score"
        ]

        ax = comparison.plot(
            kind="bar",
            figsize=(9, 5),
            width=0.7
        )

        ax.set_title(f"Desempenho da classe: {class_name.capitalize()}")
        ax.set_xlabel("Modelos")
        ax.set_ylabel("Pontuação")
        #ax.set_ylim(0.90, 1.00)
        ax.legend(title="Métricas")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.4f",
                padding=3,
                fontsize=9
            )

        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = Path(
            f"docs/images/{class_name}_comparison.png"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.show()

        print(f"\nGráfico salvo em: {output_path}")


if __name__ == "__main__":
    visualize_models()
    visualize_class_metrics()