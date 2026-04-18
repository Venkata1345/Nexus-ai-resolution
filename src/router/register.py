"""Promote the best embedding-based MLflow run to the Model Registry.

Picks the embedding run with the highest val_f1_weighted that has
`model_logged=true`, registers its model artifact under
settings.mlflow_registered_model_name, and assigns the @production alias.
The agent's router loads via this alias, so promoting a new model is a
single command — no router code change needed.
"""

from __future__ import annotations

import mlflow

from src.config import settings


def register_best() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=("tags.feature_type = 'embeddings' AND tags.model_logged = 'true'"),
        order_by=["metrics.val_f1_weighted DESC"],
    )
    if runs.empty:
        raise RuntimeError("No embedding runs with a logged model found.")

    best = runs.iloc[0]
    run_id = best.run_id
    val_f1 = best.get("metrics.val_f1_weighted") or best.get("metrics.final_val_f1_weighted")
    print(
        f"[register] Best run: {best.get('tags.mlflow.runName', '?')} "
        f"({run_id[:8]})  val_f1={val_f1}"
    )

    model_uri = f"runs:/{run_id}/{settings.mlflow_embedding_model_artifact_name}"
    mv = mlflow.register_model(model_uri=model_uri, name=settings.mlflow_registered_model_name)
    print(f"[register] Registered as {settings.mlflow_registered_model_name} version {mv.version}")

    client.set_registered_model_alias(
        name=settings.mlflow_registered_model_name,
        alias=settings.mlflow_production_alias,
        version=mv.version,
    )
    print(f"[register] Alias @{settings.mlflow_production_alias} -> version {mv.version}")


if __name__ == "__main__":
    register_best()
