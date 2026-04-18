"""Train the sentence-transformer + XGBoost intent classifier (baseline config).

Encodes the support instructions into dense 384-d embeddings using
`all-MiniLM-L6-v2`, trains XGBoost with the baseline hyperparameters from
Settings, and logs the run to MLflow under run_name="embeddings_xgb" for
side-by-side comparison with the TF-IDF baseline.

Uses the shared embedding cache under data/processed/, so the first run
pays the encoding cost once and subsequent runs (and Optuna trials) are
fast.
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.router._features import get_embeddings

LABEL_COL = "intent"


def train_embeddings() -> None:
    print("[train_emb] Loading splits...")
    train_df = pd.read_csv(settings.train_csv_path)
    val_df = pd.read_csv(settings.val_csv_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df[LABEL_COL], val_df[LABEL_COL]]))
    y_train = label_encoder.transform(train_df[LABEL_COL])
    y_val = label_encoder.transform(val_df[LABEL_COL])

    X_train = get_embeddings("train")
    X_val = get_embeddings("val")

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, settings.label_encoder_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="embeddings_xgb"):
        mlflow.set_tag("feature_type", "embeddings")
        mlflow.set_tag("classifier", "xgboost")
        mlflow.set_tag("encoder", settings.embedding_model_name)

        params = {
            "objective": "multi:softmax",
            "num_class": len(label_encoder.classes_),
            "max_depth": settings.xgb_max_depth,
            "learning_rate": settings.xgb_learning_rate,
            "n_estimators": settings.xgb_n_estimators,
        }
        params_for_log = {
            **params,
            "embedding_model": settings.embedding_model_name,
            "embedding_dim": X_train.shape[1],
        }
        mlflow.log_params(params_for_log)

        print("[train_emb] Training XGBoost on embedded features...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        print("[train_emb] Evaluating on val split...")
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds, average="weighted")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_weighted", f1)

        mlflow.xgboost.log_model(model, settings.mlflow_embedding_model_artifact_name)

        print(f"[train_emb] Done. val_accuracy={acc:.4f}  val_f1={f1:.4f}")


if __name__ == "__main__":
    train_embeddings()
