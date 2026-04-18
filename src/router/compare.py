"""Print the head-to-head comparison of TF-IDF vs embedding pipelines.

Pulls metrics directly from MLflow so the table reflects the latest runs.
"""

from __future__ import annotations

import mlflow
import pandas as pd

from src.config import settings


def compare() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_logged = 'true'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError("No model-logged runs found in MLflow.")

    columns = [
        "tags.mlflow.runName",
        "tags.feature_type",
        "tags.tuning",
        "metrics.val_accuracy",
        "metrics.val_f1_weighted",
    ]
    available = [c for c in columns if c in runs.columns]
    df = runs[available].copy()
    df.columns = [
        c.replace("tags.mlflow.runName", "run").replace("tags.", "").replace("metrics.", "")
        for c in df.columns
    ]
    df = df.sort_values("val_f1_weighted", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print()
    print("=== TF-IDF vs Embeddings -- model-logged runs ===")
    print(df.to_string(index=False))
    print()


if __name__ == "__main__":
    compare()
