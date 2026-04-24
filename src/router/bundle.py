"""Export a self-contained inference bundle for deployed containers.

Reads the currently-promoted @production model from the MLflow registry
and writes a flat folder that doesn't depend on MLflow at runtime:

    bundle/
      xgb_model.ubj        XGBoost classifier (UBJ binary, ~5 MB)
      label_encoder.pkl    sklearn LabelEncoder
      kb/                  ChromaDB persistent store (copied from data/kb)
      metadata.json        version + metrics for the UI's model panel

The deployed image COPYs ./bundle/ in and the router loads from it.
Repeat this step every time a new model is promoted.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import mlflow

from src.config import settings


def _fetch_production_version() -> Any:
    """Return the MLflow ModelVersion currently aliased @production."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    try:
        return client.get_model_version_by_alias(
            settings.mlflow_registered_model_name, settings.mlflow_production_alias
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not find @{settings.mlflow_production_alias} model. "
            "Run `python -m src.router.register` first."
        ) from exc


def bundle() -> None:
    mv = _fetch_production_version()
    print(
        f"[bundle] Exporting {mv.name} v{mv.version} (@{settings.mlflow_production_alias}) "
        f"from run {mv.run_id[:8] if mv.run_id else '?'}"
    )

    out = settings.bundle_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1) XGBoost model -> save as portable UBJ binary.
    model_uri = f"models:/{mv.name}@{settings.mlflow_production_alias}"
    loaded = mlflow.xgboost.load_model(model_uri)
    model_path = out / "xgb_model.ubj"
    loaded.save_model(str(model_path))
    print(f"[bundle] Wrote model -> {model_path}")

    # 2) LabelEncoder — already a flat pickle, just copy.
    if not settings.label_encoder_path.exists():
        raise RuntimeError(
            f"Missing {settings.label_encoder_path}. "
            "Run the training pipeline first (train_embeddings.py)."
        )
    shutil.copy(settings.label_encoder_path, out / "label_encoder.pkl")
    print(f"[bundle] Copied label_encoder -> {out / 'label_encoder.pkl'}")

    # 3) ChromaDB KB — directory copy.
    if not settings.kb_dir.exists():
        raise RuntimeError(f"Missing {settings.kb_dir}. Run `python -m src.agents.kb_build` first.")
    shutil.copytree(settings.kb_dir, out / "kb")
    print(f"[bundle] Copied KB -> {out / 'kb'}")

    # 4) Metadata for the UI's model panel.
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    metrics: dict[str, float] = {}
    if mv.run_id:
        run = client.get_run(mv.run_id)
        metrics = {k: v for k, v in run.data.metrics.items() if isinstance(v, (int, float))}

    metadata = {
        "registered_name": mv.name,
        "version": str(mv.version),
        "alias": settings.mlflow_production_alias,
        "run_id": mv.run_id,
        "metrics": metrics,
        "encoder_model": settings.embedding_model_name,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[bundle] Wrote metadata -> {out / 'metadata.json'}")

    print(f"[bundle] Done. Bundle size: approx {_dir_size_mb(out):.1f} MB")


def _dir_size_mb(path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


if __name__ == "__main__":
    bundle()
