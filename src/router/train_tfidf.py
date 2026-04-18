"""Train the TF-IDF + XGBoost baseline intent classifier.

Reads the pre-computed train/val splits from data/processed/, fits a TF-IDF
vectorizer on train ONLY (to avoid leakage), trains XGBoost on the resulting
features, and logs the run to MLflow under run_name="tfidf_xgb" so it can be
compared directly against the embedding-based pipeline.

The test split is intentionally NOT touched here -- it stays reserved for a
final unbiased evaluation in a later step.
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import settings

TEXT_COL = "instruction"
LABEL_COL = "intent"


def train_tfidf() -> None:
    print("[train_tfidf] Loading splits...")
    train_df = pd.read_csv(settings.train_csv_path)
    val_df = pd.read_csv(settings.val_csv_path)

    # Fit label encoder on the union of train+val labels so any label the
    # model might predict on val is already known.
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_df[LABEL_COL], val_df[LABEL_COL]]))
    y_train = label_encoder.transform(train_df[LABEL_COL])
    y_val = label_encoder.transform(val_df[LABEL_COL])

    # Fit TF-IDF on train ONLY. Vocabulary and IDF weights must not see val.
    print("[train_tfidf] Fitting TF-IDF on train split...")
    vectorizer = TfidfVectorizer(max_features=settings.tfidf_max_features)
    X_train = vectorizer.fit_transform(train_df[TEXT_COL])
    X_val = vectorizer.transform(val_df[TEXT_COL])

    # Persist preprocessors so the agent can reproduce features at inference.
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, settings.vectorizer_path)
    joblib.dump(label_encoder, settings.label_encoder_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="tfidf_xgb"):
        mlflow.set_tag("feature_type", "tfidf")
        mlflow.set_tag("classifier", "xgboost")
        mlflow.set_tag("model_logged", "true")

        params = {
            "objective": "multi:softmax",
            "num_class": len(label_encoder.classes_),
            "max_depth": settings.xgb_max_depth,
            "learning_rate": settings.xgb_learning_rate,
            "n_estimators": settings.xgb_n_estimators,
        }
        params_for_log = {**params, "tfidf_max_features": settings.tfidf_max_features}
        mlflow.log_params(params_for_log)

        print("[train_tfidf] Training XGBoost...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        print("[train_tfidf] Evaluating on val split...")
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds, average="weighted")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_weighted", f1)

        mlflow.xgboost.log_model(model, settings.mlflow_model_artifact_name)

        print(f"[train_tfidf] Done. val_accuracy={acc:.4f}  val_f1={f1:.4f}")


if __name__ == "__main__":
    train_tfidf()
