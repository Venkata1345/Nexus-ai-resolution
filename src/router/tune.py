"""Optuna-driven hyperparameter search over XGBoost on MiniLM embeddings.

Runs N trials (see settings.optuna_n_trials). Each trial:
  1. Samples a hyperparameter config from the search space
  2. Trains XGBoost on the cached train embeddings
  3. Scores on the val embeddings (weighted F1)
  4. Logs itself as a nested MLflow run

After the study finishes, re-train on train with the best config, evaluate on
val, and log that as the canonical model artifact on the parent run so the
agent's router picks it up via the `feature_type=embeddings` tag.
"""

from __future__ import annotations

import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.router._features import get_embeddings

LABEL_COL = "intent"


def _load_label_encoder_and_targets():
    """Build a LabelEncoder from train+val labels and return encoded targets."""
    train_df = pd.read_csv(settings.train_csv_path)
    val_df = pd.read_csv(settings.val_csv_path)

    le = LabelEncoder()
    le.fit(pd.concat([train_df[LABEL_COL], val_df[LABEL_COL]]))
    y_train = le.transform(train_df[LABEL_COL])
    y_val = le.transform(val_df[LABEL_COL])
    return le, y_train, y_val


def _build_objective(X_train, y_train, X_val, y_val, num_class):
    """Create the Optuna objective function, closing over the feature matrices."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softmax",
            "num_class": num_class,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": settings.random_seed,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="weighted")

        # Log this trial as a child run so it shows up nested in MLflow.
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.set_tag("feature_type", "embeddings")
            mlflow.set_tag("classifier", "xgboost")
            mlflow.set_tag("optuna_trial", str(trial.number))
            mlflow.log_params(params)
            mlflow.log_metric("val_f1_weighted", val_f1)

        return val_f1

    return objective


def tune() -> None:
    print("[tune] Preparing features (warm-caches on first run, fast afterwards)...")
    X_train = get_embeddings("train")
    X_val = get_embeddings("val")

    label_encoder, y_train, y_val = _load_label_encoder_and_targets()
    num_class = len(label_encoder.classes_)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="optuna_study_embeddings_xgb"):
        mlflow.set_tag("feature_type", "embeddings")
        mlflow.set_tag("classifier", "xgboost")
        mlflow.set_tag("tuning", "optuna")
        mlflow.log_param("n_trials", settings.optuna_n_trials)
        mlflow.log_param("sampler", "TPE")
        mlflow.log_param("optimization_direction", "maximize val_f1_weighted")

        # TPE sampler, seeded for reproducibility.
        sampler = optuna.samplers.TPESampler(seed=settings.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = _build_objective(X_train, y_train, X_val, y_val, num_class)
        study.optimize(objective, n_trials=settings.optuna_n_trials, show_progress_bar=False)

        print(f"[tune] Best val_f1_weighted: {study.best_value:.4f}")
        print(f"[tune] Best params: {study.best_params}")

        # Log best params (prefixed) and best score on the parent run.
        mlflow.log_metric("best_val_f1_weighted", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Retrain on train with the winning params and log as the canonical artifact.
        best_params = {
            "objective": "multi:softmax",
            "num_class": num_class,
            "random_state": settings.random_seed,
            **study.best_params,
        }
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

        # Sanity: the retrained model should match the best trial's score.
        final_preds = final_model.predict(X_val)
        final_f1 = f1_score(y_val, final_preds, average="weighted")
        mlflow.log_metric("final_val_f1_weighted", final_f1)
        print(f"[tune] Retrain val_f1_weighted: {final_f1:.4f}")

        mlflow.xgboost.log_model(final_model, settings.mlflow_embedding_model_artifact_name)


if __name__ == "__main__":
    tune()
