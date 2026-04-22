"""Build the RAG knowledge base from the Bitext support dataset.

De-duplicates by (intent, response) so we don't index 100 near-identical
copies of the same canonical answer, embeds the user-facing `instruction`
column with the same MiniLM encoder used for classification (shared
geometry), and persists everything as a Chroma collection under
settings.kb_dir.
"""

from __future__ import annotations

import contextlib

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import settings

INSTRUCTION_COL = "instruction"
INTENT_COL = "intent"
RESPONSE_COL = "response"
CATEGORY_COL = "category"


def build_kb() -> None:
    print(f"[kb] Loading {settings.raw_data_path}")
    df = pd.read_csv(settings.raw_data_path)
    df = df.dropna(subset=[INSTRUCTION_COL, INTENT_COL, RESPONSE_COL])

    # One KB entry per (intent, response). Keep first instruction as
    # the representative question for that answer.
    df = df.drop_duplicates(subset=[INTENT_COL, RESPONSE_COL]).reset_index(drop=True)
    print(f"[kb] Deduped to {len(df)} unique (intent, response) pairs")

    print(f"[kb] Loading encoder: {settings.embedding_model_name}")
    encoder = SentenceTransformer(settings.embedding_model_name)

    print("[kb] Embedding instructions...")
    embeddings = encoder.encode(
        df[INSTRUCTION_COL].tolist(),
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.kb_dir))

    # Fresh collection on every rebuild so repeated runs don't duplicate ids.
    with contextlib.suppress(Exception):
        client.delete_collection(settings.kb_collection_name)
    collection = client.create_collection(
        name=settings.kb_collection_name,
        metadata={"hnsw:space": "cosine"},  # matches our L2-normalized encoder
    )

    ids = [f"faq_{i}" for i in range(len(df))]
    metadatas = [
        {
            "intent": row[INTENT_COL],
            "category": row[CATEGORY_COL] if CATEGORY_COL in df.columns else "",
            "response": row[RESPONSE_COL],
        }
        for _, row in df.iterrows()
    ]

    # Chroma caps a single add() at ~5000 rows. Chunk to stay under the limit.
    documents = df[INSTRUCTION_COL].tolist()
    embeddings_list = embeddings.tolist()
    CHUNK = 4000
    for i in range(0, len(ids), CHUNK):
        collection.add(
            ids=ids[i : i + CHUNK],
            embeddings=embeddings_list[i : i + CHUNK],
            documents=documents[i : i + CHUNK],
            metadatas=metadatas[i : i + CHUNK],
        )

    print(
        f"[kb] Wrote collection {settings.kb_collection_name!r} with "
        f"{collection.count()} entries to {settings.kb_dir}"
    )


if __name__ == "__main__":
    build_kb()
