"""Router node: classify the user's latest message into an intent + confidence."""

from __future__ import annotations

from functools import lru_cache

import joblib
import mlflow
from sentence_transformers import SentenceTransformer

from src.agents.state import NexusState
from src.config import settings


def _load_production_model():
    """Load the model currently aliased @production in the MLflow Model Registry.

    Promotion to @production is done by `python -m src.router.register`. The
    agent never references run IDs directly, so swapping in a new model is a
    one-liner and no code changes here.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.mlflow_registered_model_name}@{settings.mlflow_production_alias}"
    try:
        return mlflow.xgboost.load_model(uri)
    except mlflow.exceptions.MlflowException as exc:
        raise RuntimeError(
            f"No model aliased @{settings.mlflow_production_alias} for "
            f"{settings.mlflow_registered_model_name!r}. "
            f"Train and register one first: "
            f"`python -m src.router.tune && python -m src.router.register`."
        ) from exc


@lru_cache(maxsize=1)
def _get_deps():
    """Load encoder + label encoder + classifier once, reuse forever.

    Pulled out of module scope so importing this module doesn't touch disk or
    MLflow -- tests can patch `_predict` or inject fakes without dragging in
    the real heavy dependencies.
    """
    label_encoder = joblib.load(settings.label_encoder_path)
    encoder = SentenceTransformer(settings.embedding_model_name)
    intent_model = _load_production_model()
    return encoder, intent_model, label_encoder


def _predict(text: str, encoder, intent_model, label_encoder) -> tuple[str, float]:
    """Pure prediction helper: text -> (intent, confidence). Easy to unit-test."""
    vec = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    probs = intent_model.predict_proba(vec)[0]
    top_idx = int(probs.argmax())
    intent = label_encoder.inverse_transform([top_idx])[0]
    return str(intent), float(probs[top_idx])


def predict_intent_node(state: NexusState):
    """Predict intent + confidence from the latest message and store both in state."""
    encoder, intent_model, label_encoder = _get_deps()

    messages = state.get("messages", [])
    latest_user_text = messages[-1].content
    intent, confidence = _predict(latest_user_text, encoder, intent_model, label_encoder)

    print(f"\n[Router Node] Analyzed ticket. Intent='{intent}'  confidence={confidence:.3f}")
    return {"intent": intent, "intent_confidence": confidence}


def route_after_prediction(state: NexusState):
    """Conditional edge: pick the next node based on the predicted intent."""
    intent = state.get("intent")
    action_intents = ["track_order", "cancel_order", "get_refund", "check_invoices"]

    if intent in action_intents:
        print("[Edge] Routing to Action Node (Database Required)...")
        return "action_node"
    print("[Edge] Routing to RAG Node (Documentation Required)...")
    return "rag_node"
