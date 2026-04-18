"""Final evaluation on the held-out test set.

This is the ONE allowed touch of test.csv. We:
  1. Encode test text with the same MiniLM encoder (cached)
  2. Load the latest tuned embedding model from MLflow
  3. Compute aggregate + per-class metrics
  4. Render a confusion matrix
  5. Benchmark single-request and batch latency
  6. Log everything (metrics + figures + reports) to a dedicated MLflow run

The numbers reported here are the unbiased ones safe to put on a README,
resume, or in a blog post.
"""

from __future__ import annotations

import time

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import settings
from src.router._features import get_embeddings

TEXT_COL = "instruction"
LABEL_COL = "intent"

# Latency benchmark configuration.
N_LATENCY_SAMPLES = 100
N_WARMUP_CALLS = 5
BATCH_SIZE = 64


def _load_latest_embedding_model():
    """Pull the most recent embedding-tagged model from MLflow.

    Excludes Optuna trials (children, no model artifact) and evaluation runs
    (also no model artifact). Concept #6 (Model Registry) replaces this
    heuristic with a proper alias lookup.
    """
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.feature_type = 'embeddings' AND tags.model_logged = 'true'"
        ),
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError("No embedding-tagged MLflow run with a model artifact found.")
    run_id = runs.iloc[0].run_id
    return mlflow.xgboost.load_model(
        f"runs:/{run_id}/{settings.mlflow_embedding_model_artifact_name}"
    )


def _benchmark_single_latency(
    texts: list[str], encoder: SentenceTransformer, model
) -> dict[str, float]:
    """Time end-to-end (encode + predict) for individual requests."""
    rng = np.random.default_rng(settings.random_seed)
    sample_idx = rng.choice(len(texts), size=N_LATENCY_SAMPLES + N_WARMUP_CALLS, replace=False)

    # Warmup — the first calls pay JIT and allocator costs we shouldn't include.
    for i in sample_idx[:N_WARMUP_CALLS]:
        emb = encoder.encode(
            [texts[i]], convert_to_numpy=True, normalize_embeddings=True
        )
        model.predict(emb)

    timings_ms: list[float] = []
    for i in sample_idx[N_WARMUP_CALLS:]:
        start = time.perf_counter()
        emb = encoder.encode(
            [texts[i]], convert_to_numpy=True, normalize_embeddings=True
        )
        model.predict(emb)
        timings_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(timings_ms)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
    }


def _benchmark_batch_throughput(
    texts: list[str], encoder: SentenceTransformer, model
) -> dict[str, float]:
    """Measure tickets/sec when processing a batch of `BATCH_SIZE` at once."""
    batch = texts[:BATCH_SIZE]
    # Warmup
    encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)

    start = time.perf_counter()
    embs = encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    model.predict(embs)
    elapsed_s = time.perf_counter() - start
    return {
        "batch_total_ms": elapsed_s * 1000,
        "throughput_tickets_per_sec": BATCH_SIZE / elapsed_s,
    }


def evaluate() -> None:
    print("[evaluate] Loading test set...")
    test_df = pd.read_csv(settings.test_csv_path)
    label_encoder = joblib.load(settings.label_encoder_path)
    y_test = label_encoder.transform(test_df[LABEL_COL])

    print("[evaluate] Embedding test set (or loading cache)...")
    X_test = get_embeddings("test")

    print("[evaluate] Loading latest tuned embedding model...")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    model = _load_latest_embedding_model()

    # ---------- Predictions ----------
    print("[evaluate] Scoring test set...")
    y_pred = model.predict(X_test)

    # ---------- Aggregate metrics ----------
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    test_f1_macro = f1_score(y_test, y_pred, average="macro")

    print()
    print(f"[evaluate] test_accuracy   = {test_accuracy:.4f}")
    print(f"[evaluate] test_f1_weighted = {test_f1_weighted:.4f}")
    print(f"[evaluate] test_f1_macro    = {test_f1_macro:.4f}")

    # ---------- Per-class report ----------
    class_names = list(label_encoder.classes_)
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print()
    print("[evaluate] === Per-class classification report ===")
    print(report_str)

    # ---------- Confusion matrix figure ----------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(
        f"Confusion Matrix (Test set) -- weighted F1 = {test_f1_weighted:.4f}",
        fontsize=14,
    )
    plt.tight_layout()

    # ---------- Latency benchmarks ----------
    print()
    print("[evaluate] Running latency benchmarks...")
    encoder = SentenceTransformer(settings.embedding_model_name)
    test_texts = test_df[TEXT_COL].tolist()

    single_stats = _benchmark_single_latency(test_texts, encoder, model)
    batch_stats = _benchmark_batch_throughput(test_texts, encoder, model)

    print(f"[evaluate] Single-request mean: {single_stats['latency_ms_mean']:.2f} ms")
    print(f"[evaluate] Single-request p50/p95/p99: "
          f"{single_stats['latency_ms_p50']:.2f} / "
          f"{single_stats['latency_ms_p95']:.2f} / "
          f"{single_stats['latency_ms_p99']:.2f} ms")
    print(f"[evaluate] Batch-{BATCH_SIZE} throughput: "
          f"{batch_stats['throughput_tickets_per_sec']:.1f} tickets/sec")

    # ---------- MLflow logging ----------
    with mlflow.start_run(run_name="test_evaluation_embeddings_xgb"):
        mlflow.set_tag("feature_type", "embeddings")
        mlflow.set_tag("classifier", "xgboost")
        mlflow.set_tag("evaluation", "test_set_final")

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        for k, v in single_stats.items():
            mlflow.log_metric(k, v)
        for k, v in batch_stats.items():
            mlflow.log_metric(k, v)

        mlflow.log_text(report_str, "classification_report.txt")
        mlflow.log_figure(fig, "confusion_matrix.png")

    plt.close(fig)
    print("\n[evaluate] Done. Artifacts logged to MLflow.")


if __name__ == "__main__":
    evaluate()
