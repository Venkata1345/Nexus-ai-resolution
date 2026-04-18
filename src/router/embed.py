"""Materialize embedding caches for all three splits.

Exists as a dedicated DVC stage so the embedding step has explicit declared
outputs (`*_embeddings.npy`) that downstream training/tuning stages can
depend on. Without this, DVC couldn't tell when embeddings need re-doing.
"""

from __future__ import annotations

from src.router._features import get_embeddings


def main() -> None:
    for split in ("train", "val", "test"):
        get_embeddings(split)


if __name__ == "__main__":
    main()
