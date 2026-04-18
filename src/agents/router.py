import joblib
import mlflow

from src.agents.state import NexusState
from src.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

vectorizer = joblib.load(settings.vectorizer_path)
label_encoder = joblib.load(settings.label_encoder_path)


def load_latest_model():
    """Fetch the most recent TF-IDF run from MLflow.

    The experiment also contains embedding-based runs with a different
    artifact name, so we filter by the `feature_type=tfidf` tag to guarantee
    we load a run that actually has the expected artifact.
    """
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)

    df_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.feature_type = 'tfidf'",
        order_by=["start_time DESC"],
    )
    if df_runs.empty:
        raise RuntimeError(
            "No MLflow runs found with feature_type=tfidf. "
            "Train the baseline first: `python -m src.router.train_tfidf`."
        )
    latest_run_id = df_runs.iloc[0].run_id

    return mlflow.xgboost.load_model(f"runs:/{latest_run_id}/{settings.mlflow_model_artifact_name}")


intent_model = load_latest_model()


def predict_intent_node(state: NexusState):
    """Predict the customer's intent from the latest message and store it in state."""
    messages = state.get("messages", [])
    latest_user_text = messages[-1].content

    vec_input = vectorizer.transform([latest_user_text])
    prediction_encoded = intent_model.predict(vec_input)
    predicted_intent = label_encoder.inverse_transform(prediction_encoded)[0]

    print(f"\n[Router Node] Analyzed ticket. Predicted Intent: '{predicted_intent}'")
    return {"intent": predicted_intent}


def route_after_prediction(state: NexusState):
    """Conditional edge: pick the next node based on the predicted intent."""
    intent = state.get("intent")
    action_intents = ["track_order", "cancel_order", "get_refund", "check_invoices"]

    if intent in action_intents:
        print("[Edge] Routing to Action Node (Database Required)...")
        return "action_node"
    print("[Edge] Routing to RAG Node (Documentation Required)...")
    return "rag_node"
