"""RAG retrieval node: fetch top-k canonical responses for the user's query."""

from __future__ import annotations

import chromadb
from langchain_core.messages import SystemMessage
from sentence_transformers import SentenceTransformer

from src.agents.state import NexusState
from src.config import settings

# Load once at module import — same pattern as the classifier encoder.
_encoder = SentenceTransformer(settings.embedding_model_name)
_client = chromadb.PersistentClient(path=str(settings.kb_dir))
_collection = _client.get_collection(settings.kb_collection_name)


def retrieve_knowledge_node(state: NexusState):
    """Find the top-k most similar FAQ entries and inject them into state as context."""
    print("\n[Retriever] Searching knowledge base...")

    latest_user_text = state["messages"][-1].content
    query_emb = _encoder.encode(
        [latest_user_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = _collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=settings.kb_top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Build a single SystemMessage with the retrieved Q/A pairs so the LLM
    # sees real canonical answers it can quote from or paraphrase.
    lines = ["KNOWLEDGE BASE RESULTS (use these to ground your answer):"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists, strict=True), start=1):
        sim = 1.0 - float(dist)  # cosine distance -> similarity
        lines.append(
            f"\n[{i}] (intent={meta['intent']}, similarity={sim:.3f})\n"
            f"    Q: {doc}\n"
            f"    A: {meta['response']}"
        )
    note = SystemMessage(content="\n".join(lines))

    top_sim = 1.0 - float(dists[0]) if dists else 0.0
    print(f"[Retriever] Top-{settings.kb_top_k} retrieved. Best similarity: {top_sim:.3f}")

    return {"messages": [note]}
