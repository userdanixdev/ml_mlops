
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_randomized_parameters():

    results_path = Path("reports/optimization/randomized_search_results.csv")
    results = pd.read_csv(results_path)

    parameter_columns = [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features"
    ]

    parameters = results[parameter_columns].copy()

    parameters.columns = [
        "N Estimators",
        "Max Depth",
        "Min Samples Split",
        "Min Samples Leaf",
        "Max Features"
    ]

    parameters = parameters.fillna("None").astype(str)

    best_index = results["mean_test_score"].idxmax()
    best_test = best_index + 1

    parameters.index = [
        f"Teste {i + 1}" for i in range(len(parameters))
    ]

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        parameters.apply(lambda column: pd.factorize(column)[0]),
        annot=parameters,
        fmt="",
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor="white"
    )

    plt.title(
        f"RandomizedSearchCV - Configurações dos Hiperparâmetros\n"
        f"Melhor combinação: Teste {best_test}"
    )

    plt.xlabel("Hiperparâmetros")
    plt.ylabel("Combinações avaliadas")

    plt.tight_layout()

    output_path = Path(
        "reports/optimization/randomized_search_parameters.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráfico salvo em: {output_path}")
    print(f"Melhor teste: {best_test}")
    print(f"Melhor F1-score macro: {results.loc[best_index, 'mean_test_score']:.4f}")


if __name__ == "__main__":
    visualize_randomized_parameters()
