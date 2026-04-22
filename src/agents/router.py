import joblib
import mlflow
from sentence_transformers import SentenceTransformer

from src.agents.state import NexusState
from src.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

# Load preprocessors + encoder once at module import so inference is hot.
label_encoder = joblib.load(settings.label_encoder_path)
encoder = SentenceTransformer(settings.embedding_model_name)


def load_production_model():
    """Load the model currently aliased @production in the MLflow Model Registry.

    Promotion to @production is done by `python -m src.router.register`. The
    agent never references run IDs directly, so swapping in a new model is a
    one-liner and no code changes here.
    """
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


intent_model = load_production_model()


def predict_intent_node(state: NexusState):
    """Predict intent + confidence from the latest message and store both in state."""
    messages = state.get("messages", [])
    latest_user_text = messages[-1].content

    # Encode with the same normalization used during training so features match.
    vec_input = encoder.encode(
        [latest_user_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # predict_proba returns per-class probabilities; we take the argmax and
    # its probability as the "confidence" the escalation gate checks.
    probs = intent_model.predict_proba(vec_input)[0]
    top_idx = int(probs.argmax())
    predicted_intent = label_encoder.inverse_transform([top_idx])[0]
    confidence = float(probs[top_idx])

    print(
        f"\n[Router Node] Analyzed ticket. Intent='{predicted_intent}'  confidence={confidence:.3f}"
    )
    return {"intent": predicted_intent, "intent_confidence": confidence}


def route_after_prediction(state: NexusState):
    """Conditional edge: pick the next node based on the predicted intent."""
    intent = state.get("intent")
    action_intents = ["track_order", "cancel_order", "get_refund", "check_invoices"]

    if intent in action_intents:
        print("[Edge] Routing to Action Node (Database Required)...")
        return "action_node"
    print("[Edge] Routing to RAG Node (Documentation Required)...")
    return "rag_node"
