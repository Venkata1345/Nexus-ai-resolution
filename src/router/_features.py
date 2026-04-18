"""Shared feature-extraction helpers for the embedding pipeline.

Extracted out of train_embeddings.py so both the training entrypoint and the
hyperparameter tuner can reuse the same encoding logic and the same on-disk
cache. Embedding is the slow step; without caching, Optuna would re-encode
the train set on every trial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import settings

TEXT_COL = "instruction"


def encode_texts(texts: list[str], encoder: SentenceTransformer) -> np.ndarray:
    """Batch-encode strings into a (N, dim) float32 matrix, normalized."""
    return encoder.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def _embed_or_load(csv_path: Path, cache_path: Path, encoder: SentenceTransformer) -> np.ndarray:
    """Load cached embeddings if present, otherwise embed and save."""
    if cache_path.exists():
        print(f"[features] Loading cached embeddings: {cache_path.name}")
        return np.load(cache_path)

    print(f"[features] Encoding {csv_path.name} (no cache)...")
    df = pd.read_csv(csv_path)
    embeddings = encode_texts(df[TEXT_COL].tolist(), encoder)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"[features] Cached -> {cache_path}")
    return embeddings


def get_embeddings(split: str) -> np.ndarray:
    """Public entry: get the embedding matrix for 'train', 'val', or 'test'."""
    paths = {
        "train": (settings.train_csv_path, settings.train_embeddings_path),
        "val": (settings.val_csv_path, settings.val_embeddings_path),
        "test": (settings.test_csv_path, settings.test_embeddings_path),
    }
    if split not in paths:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(paths)}")

    csv_path, cache_path = paths[split]
    if cache_path.exists():
        print(f"[features] Loading cached embeddings: {cache_path.name}")
        return np.load(cache_path)

    encoder = SentenceTransformer(settings.embedding_model_name)
    return _embed_or_load(csv_path, cache_path, encoder)
