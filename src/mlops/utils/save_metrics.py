
import pandas as pd
from pathlib import Path

def save_metrics(model_name, accuracy, classification_metrics):
    metrics_path = Path("reports/model_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "model": model_name,
        "accuracy": accuracy
    }

    # Organiza as métricas de cada classe e as médias
    for category, values in classification_metrics.items():

        if isinstance(values, dict):
            for metric_name, metric_value in values.items():
                column_name = f"{category}_{metric_name}"
                metrics_data[column_name] = metric_value

    new_metrics = pd.DataFrame([metrics_data])

    # Mantém apenas o resultado mais recente de cada modelo
    if metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        existing_metrics = existing_metrics[
            existing_metrics["model"] != model_name
        ]

        new_metrics = pd.concat(
            [existing_metrics, new_metrics],
            ignore_index=True
        )

    new_metrics.to_csv(metrics_path, index=False)

    print(f"\nMétricas salvas em: {metrics_path}")
