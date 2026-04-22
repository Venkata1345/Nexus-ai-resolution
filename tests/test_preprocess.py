"""Tests for the stratified train/val/test split."""

from __future__ import annotations

import pandas as pd
import pytest

from src.router.preprocess import _stratified_three_way_split


def _fake_dataset(per_class: int = 50, n_classes: int = 4) -> pd.DataFrame:
    rows = []
    for c in range(n_classes):
        for i in range(per_class):
            rows.append({"instruction": f"sample {c}-{i}", "intent": f"intent_{c}"})
    return pd.DataFrame(rows)


def test_split_sizes_match_configured_fractions():
    df = _fake_dataset(per_class=100, n_classes=5)  # 500 rows
    train, val, test = _stratified_three_way_split(
        df, val_fraction=0.15, test_fraction=0.15, random_seed=42
    )
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(test) / total - 0.15) < 0.02
    assert abs(len(val) / total - 0.15) < 0.02
    assert abs(len(train) / total - 0.70) < 0.02


def test_splits_are_disjoint():
    df = _fake_dataset(per_class=50, n_classes=4)
    train, val, test = _stratified_three_way_split(
        df, val_fraction=0.15, test_fraction=0.15, random_seed=42
    )
    train_keys = set(train["instruction"])
    val_keys = set(val["instruction"])
    test_keys = set(test["instruction"])
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)


def test_every_class_present_in_every_split():
    df = _fake_dataset(per_class=100, n_classes=6)
    train, val, test = _stratified_three_way_split(
        df, val_fraction=0.15, test_fraction=0.15, random_seed=42
    )
    all_classes = set(df["intent"].unique())
    assert set(train["intent"].unique()) == all_classes
    assert set(val["intent"].unique()) == all_classes
    assert set(test["intent"].unique()) == all_classes


def test_stratification_preserves_class_ratios():
    # Deliberately imbalanced: class_0 is 5x more common than class_1.
    df = pd.DataFrame(
        [{"instruction": f"a{i}", "intent": "class_0"} for i in range(500)]
        + [{"instruction": f"b{i}", "intent": "class_1"} for i in range(100)]
    )
    train, val, test = _stratified_three_way_split(
        df, val_fraction=0.2, test_fraction=0.2, random_seed=7
    )
    for split in (train, val, test):
        frac_0 = (split["intent"] == "class_0").mean()
        # Original ratio is 500/600 ~= 0.833. Allow 3pp slack for rounding.
        assert abs(frac_0 - 500 / 600) < 0.03


def test_split_is_deterministic_under_same_seed():
    df = _fake_dataset(per_class=80, n_classes=3)
    a = _stratified_three_way_split(df, val_fraction=0.15, test_fraction=0.15, random_seed=42)
    b = _stratified_three_way_split(df, val_fraction=0.15, test_fraction=0.15, random_seed=42)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(
            x.sort_index().reset_index(drop=True), y.sort_index().reset_index(drop=True)
        )


@pytest.mark.parametrize("seed", [1, 13, 99])
def test_different_seeds_produce_different_splits(seed: int):
    df = _fake_dataset(per_class=80, n_classes=3)
    base = _stratified_three_way_split(df, val_fraction=0.15, test_fraction=0.15, random_seed=42)
    other = _stratified_three_way_split(df, val_fraction=0.15, test_fraction=0.15, random_seed=seed)
    base_train = set(base[0]["instruction"])
    other_train = set(other[0]["instruction"])
    # Not identical.
    assert base_train != other_train
