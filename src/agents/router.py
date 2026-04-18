import joblib
import mlflow

from src.agents.state import NexusState

# 1. Connect to our local MLOps infrastructure
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load the NLP preprocessors we saved during training
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def load_latest_model():
    """Dynamically fetches the most recently trained model from MLflow."""
    experiment = mlflow.get_experiment_by_name("nexus_intent_classification")

    # Search for the latest run so we don't have to hardcode a Run ID
    df_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )
    latest_run_id = df_runs.iloc[0].run_id

    # Load the artifact
    return mlflow.xgboost.load_model(f"runs:/{latest_run_id}/xgboost_intent_model")


# Load the model into memory when the app starts
intent_model = load_latest_model()


def predict_intent_node(state: NexusState):
    """
    NODE: Reads the customer's text from the message thread, predicts the intent
    using XGBoost, and saves the result back into the shared LangGraph state.
    """
    # --- THE ARCHITECTURAL FIX ---
    # 1. Grab the list of messages from the state
    messages = state.get("messages", [])

    # 2. Extract the text content from the most recent message
    # (Since this is the first node, messages[-1] is the user's initial question)
    latest_user_text = messages[-1].content

    # 3. Vectorize and predict using the extracted text
    vec_input = vectorizer.transform([latest_user_text])
    prediction_encoded = intent_model.predict(vec_input)

    # Translate the integer back to readable text
    predicted_intent = label_encoder.inverse_transform(prediction_encoded)[0]

    print(f"\n[Router Node] Analyzed ticket. Predicted Intent: '{predicted_intent}'")

    return {"intent": predicted_intent}


def route_after_prediction(state: NexusState):
    """
    CONDITIONAL EDGE: Looks at the predicted intent and dictates the next step.
    """
    intent = state.get("intent")

    # Define which intents require database actions vs FAQ lookups
    # (We are using a few common ones from the Bitext dataset here)
    action_intents = ["track_order", "cancel_order", "get_refund", "check_invoices"]

    if intent in action_intents:
        print("[Edge] Routing to Action Node (Database Required)...")
        return "action_node"
    else:
        print("[Edge] Routing to RAG Node (Documentation Required)...")
        return "rag_node"
