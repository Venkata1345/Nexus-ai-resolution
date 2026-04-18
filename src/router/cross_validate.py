"""K-fold stratified cross-validation for the Optuna-best hyperparameters.

Purpose:
    Our Optuna study reported 99.16% val F1 on a single train/val split. That
    number is noisy — it depends on which 15% of data ended up in val. CV
    re-runs training K times with different val folds and reports mean ± std,
    so we know how stable the estimate really is and whether the Optuna
    hyperparameters overfit to one particular split.

The test set is intentionally NOT used here. It stays sacred for the final
unbiased evaluation in the next concept.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config import settings
from src.router._features import get_embeddings

LABEL_COL = "intent"

# The winning hyperparameters from the Optuna study (see MLflow run
# `optuna_study_embeddings_xgb`). Hard-coded here so this script stands on
# its own and can be re-run without re-querying MLflow. Concept #6 (Model
# Registry) will replace this with a registry lookup.
BEST_PARAMS: dict = {
    "objective": "multi:softmax",
    "max_depth": 3,
    "n_estimators": 298,
    "learning_rate": 0.10471209213501693,
    "min_child_weight": 8,
    "subsample": 0.9085081386743783,
    "colsample_bytree": 0.6296178606936361,
    "reg_alpha": 7.374385355858303e-06,
    "reg_lambda": 8.451863533931625e-08,
    # Speed: histogram algorithm + all cores. Same math, much faster.
    "tree_method": "hist",
    "n_jobs": -1,
}

N_FOLDS = 5


def cross_validate() -> None:
    print("[cv] Loading cached embeddings for train+val...")
    X_train = get_embeddings("train")
    X_val = get_embeddings("val")
    X = np.vstack([X_train, X_val])  # (train+val, 384)

    train_df = pd.read_csv(settings.train_csv_path)
    val_df = pd.read_csv(settings.val_csv_path)
    y_raw = pd.concat([train_df[LABEL_COL], val_df[LABEL_COL]]).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    params = {**BEST_PARAMS, "num_class": len(label_encoder.classes_)}

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=settings.random_seed,
    )

    fold_accs: list[float] = []
    fold_f1s: list[float] = []

    print(f"[cv] Running {N_FOLDS}-fold stratified CV...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, random_state=settings.random_seed)
        model.fit(X_fold_train, y_fold_train)

        preds = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, preds)
        f1 = f1_score(y_fold_val, preds, average="weighted")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"[cv]   fold {fold_idx}/{N_FOLDS}: acc={acc:.4f}  f1={f1:.4f}")

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))

    print()
    print("[cv] === Cross-validation summary ===")
    print(f"[cv] Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"[cv] F1:       {mean_f1:.4f} (+/- {std_f1:.4f})")

    # Log to MLflow as a single dedicated run for this analysis.
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"cv_embeddings_xgb_{N_FOLDS}fold"):
        mlflow.set_tag("feature_type", "embeddings")
        mlflow.set_tag("classifier", "xgboost")
        mlflow.set_tag("evaluation", "cross_validation")

        mlflow.log_params({**params, "n_folds": N_FOLDS})
        mlflow.log_metric("cv_mean_accuracy", mean_acc)
        mlflow.log_metric("cv_std_accuracy", std_acc)
        mlflow.log_metric("cv_mean_f1_weighted", mean_f1)
        mlflow.log_metric("cv_std_f1_weighted", std_f1)
        for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), start=1):
            mlflow.log_metric(f"fold_{i}_accuracy", a)
            mlflow.log_metric(f"fold_{i}_f1_weighted", f)


if __name__ == "__main__":
    cross_validate()
