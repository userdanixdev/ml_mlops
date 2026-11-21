import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_randomized_search():

    results_path = Path("reports/optimization/randomized_search_results.csv")
    results = pd.read_csv(results_path)
    results["test_number"] = range(1, len(results) + 1)

    results = results.sort_values(
        by="mean_test_score",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        results["test_number"],
        results["mean_test_score"],
        width=0.9,
        color="lightgray",
        edgecolor="black",
        linewidth=0.5
    )

    ax.bar_label(
    bars,
    fmt="%.4f",
    label_type="center",
    fontsize=8,
    color="black"
)

    ax.set_title("RandomizedSearchCV - F1-Score Macro por Teste")
    ax.set_xlabel("Número do teste")
    ax.set_ylabel("F1-score Macro (média da validação cruzada)")

    ax.set_xlim(
        max(0, results["mean_test_score"].min() - 0.05),
        min(1, results["mean_test_score"].max() + 0.03)
    )
    ax.set_xticks(results["test_number"])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = Path("reports/optimization/randomized_search_scores.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"\nGráfico salvo em: {output_path}")


def visualize_best_model_metrics():

    metrics_path = Path("reports/model_metrics.csv")

    metrics = pd.read_csv(metrics_path)

    best_model = metrics[
        metrics["model"] == "Random Forest Optimized"
    ]

    if best_model.empty:
        print("Modelo otimizado não encontrado no CSV.")
        return

    best_model = best_model.iloc[0]

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    metric_values = [
        best_model["accuracy"],
        best_model["macro avg_precision"],
        best_model["macro avg_recall"],
        best_model["macro avg_f1-score"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        metric_names,
        metric_values,
        width=0.6,
        color="lightgray",
        edgecolor="black",
        linewidth=0.5
    )

    ax.bar_label(
        bars,
        fmt="%.4f",
        label_type="center",
        fontsize=9,
        color="black"
    )

    ax.set_title("Random Forest Otimizado - Métricas no Teste")
    ax.set_ylabel("Pontuação")
    ax.set_ylim(
        max(0, min(metric_values) - 0.05),
        min(1, max(metric_values) + 0.03)
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = Path(
        "reports/optimization/randomized_search_best_metrics.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráfico salvo em: {output_path}")


if __name__ == "__main__":
    visualize_randomized_search()
    visualize_best_model_metrics()