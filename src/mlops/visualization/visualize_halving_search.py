
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def visualize_halving_optimization():

    # 1. Caminhos dos arquivos
    input_path = Path("reports/optimization/halving_search_results.csv")
    output_dir = Path("reports/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Carregamento dos resultados
    results = pd.read_csv(input_path)

    # 3. Agrupamento por iteração
    iteration_data = (
        results.groupby("iter")
        .agg(
            candidates=("iter", "size"),
            resources=("n_resources", "first")
        )
        .reset_index()
        .sort_values("iter")
    )

    iteration_labels = [
        f"Iteração {i + 1}"
        for i in iteration_data["iter"]
    ]

    # 4. Gráfico: redução de candidatos
    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        iteration_labels,
        iteration_data["candidates"]
    )

    plt.title("Redução de Candidatos por Iteração")
    plt.xlabel("Iterações")
    plt.ylabel("Quantidade de candidatos")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "halving_candidates_per_iteration.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # 5. Gráfico: aumento de recursos
    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        iteration_labels,
        iteration_data["resources"]
    )

    plt.title("Aumento de Recursos por Candidato")
    plt.xlabel("Iterações")
    plt.ylabel("Quantidade de amostras")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "halving_resources_per_iteration.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("\nGráficos gerados com sucesso!")
    print(output_dir / "halving_candidates_per_iteration.png")
    print(output_dir / "halving_resources_per_iteration.png")


if __name__ == "__main__":
    visualize_halving_optimization()
