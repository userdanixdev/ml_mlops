
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def visualize_halving_best_model():

    input_path = Path("reports/optimization/halving_search_results.csv")
    output_path = Path(
        "reports/optimization/halving_best_model_evaluation.png"
    )

    results = pd.read_csv(input_path)

    # Calcula a média e o melhor resultado de cada iteração
    iteration_metrics = (
        results.groupby("iter")["mean_test_score"]
        .agg(["mean", "max"])
        .reset_index()
    )

    # Identifica a iteração com o maior F1 Macro individual
    best_iteration = iteration_metrics.loc[
        iteration_metrics["max"].idxmax()
    ]

    # Identifica o melhor candidato de toda a busca
    best_candidate = results.loc[
        results["mean_test_score"].idxmax()
    ]

    colors = [
        "seagreen" if iteration == best_iteration["iter"] else "steelblue"
        for iteration in iteration_metrics["iter"]
    ]

    # Visualização
    plt.figure(figsize=(8, 4.5))

    bars = plt.bar(
        iteration_metrics["iter"].astype(str),
        iteration_metrics["mean"],
        color=colors,
        width=0.35,
        label="F1 Macro médio"
    )

    # Exibe a média de cada iteração
    for bar, value in zip(bars, iteration_metrics["mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.4f}",
            ha="center",
            fontsize=9
        )

    # Destaca o melhor candidato encontrado
    plt.axhline(
        best_candidate["mean_test_score"],
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=f"Melhor candidato: {best_candidate['mean_test_score']:.4f}"
    )

    plt.title(
        "Desempenho dos Candidatos por Iteração - HalvingGridSearchCV",
        fontsize=11
    )
    plt.xlabel("Iterações")
    plt.ylabel("F1 Macro médio")

    plt.xticks(
        range(len(iteration_metrics)),
        [f"Iteração {i + 1}" for i in iteration_metrics["iter"]],
        fontsize=9
    )

    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Legenda abaixo da área do gráfico
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nMédia de F1 Macro por iteração:")
    print(iteration_metrics.to_string(index=False))

    print(
        f"\nMelhor F1 Macro encontrado: "
        f"{best_candidate['mean_test_score']:.4f}"
    )

    print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    visualize_halving_best_model()
