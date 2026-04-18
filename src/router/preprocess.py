"""Split the raw Bitext CSV into stratified train / val / test CSVs.

Why this module exists:
    The raw dataset is a single CSV. Training needs three disjoint splits:
    - train: fit model weights
    - val:   tune hyperparameters, pick the best config (Optuna uses this)
    - test:  touched once at the end for an unbiased final metric
    Doing the split in a dedicated preprocessing step (rather than inline in
    train.py) means:
      1. Every training run uses IDENTICAL splits — no chance of accidentally
         reshuffling between Optuna trials and biasing comparisons.
      2. DVC can track the split files as pipeline outputs, so `dvc repro`
         only re-splits when raw data or split params change.
      3. Splits are inspectable on disk — you can grep `test.csv` to confirm
         no leaks into train.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings

# Columns we actually care about from the Bitext dataset.
TEXT_COL = "instruction"
LABEL_COL = "intent"


def _load_and_clean(csv_path) -> pd.DataFrame:
    """Load raw CSV and drop rows with missing text or label."""
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} rows with missing text/label.")
    return df


def _stratified_three_way_split(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 (or whatever the config says) split.

    sklearn only does a 2-way split, so we chain two calls:
      1. First split off the test set from everything else.
      2. Then split the remainder into train and val, with val_fraction
         RESCALED relative to the remaining size (not the original).
    """
    # Step 1: carve out the test set.
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df[LABEL_COL],
        random_state=random_seed,
    )

    # Step 2: from the remaining trainval, carve out val.
    # If overall val is 15%, and test already took 15%, then val needs to be
    # 15 / 85 ≈ 0.1765 of trainval so it ends up as 15% of the original.
    val_size_within_remainder = val_fraction / (1.0 - test_fraction)

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_size_within_remainder,
        stratify=trainval_df[LABEL_COL],
        random_state=random_seed,
    )

    return train_df, val_df, test_df


def _report_split_sizes(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)
    print(
        f"[preprocess] Sizes -- train: {len(train_df)} "
        f"({len(train_df) / total:.1%}) | "
        f"val: {len(val_df)} ({len(val_df) / total:.1%}) | "
        f"test: {len(test_df)} ({len(test_df) / total:.1%}) | "
        f"total: {total}"
    )

    # Sanity: every class should appear in every split.
    train_labels = set(train_df[LABEL_COL].unique())
    val_labels = set(val_df[LABEL_COL].unique())
    test_labels = set(test_df[LABEL_COL].unique())
    missing_in_val = train_labels - val_labels
    missing_in_test = train_labels - test_labels
    if missing_in_val:
        print(f"[preprocess] WARNING: classes missing from val: {missing_in_val}")
    if missing_in_test:
        print(f"[preprocess] WARNING: classes missing from test: {missing_in_test}")


def preprocess() -> None:
    """Load raw CSV, split it, and write three CSVs under processed_data_dir."""
    print(f"[preprocess] Reading {settings.raw_data_path}")
    df = _load_and_clean(settings.raw_data_path)

    train_df, val_df, test_df = _stratified_three_way_split(
        df,
        val_fraction=settings.val_fraction,
        test_fraction=settings.test_fraction,
        random_seed=settings.random_seed,
    )

    _report_split_sizes(train_df, val_df, test_df)

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(settings.train_csv_path, index=False)
    val_df.to_csv(settings.val_csv_path, index=False)
    test_df.to_csv(settings.test_csv_path, index=False)

    print(
        f"[preprocess] Wrote:\n"
        f"  train -> {settings.train_csv_path}\n"
        f"  val   -> {settings.val_csv_path}\n"
        f"  test  -> {settings.test_csv_path}"
    )


if __name__ == "__main__":
    preprocess()
