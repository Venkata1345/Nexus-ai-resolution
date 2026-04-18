import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import os
import joblib

# 1. Point MLflow to a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nexus_intent_classification")

def train_model():
    print("Loading data...")
    # Read the dataset we just downloaded
    df = pd.read_csv("data/raw/bitext_support_data.csv")
    
    X = df['instruction']
    y = df['intent']

    # 2. Text Vectorization and Label Encoding
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # Save the preprocessors for when we test the model later
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

    # 3. Model Training & MLflow Logging Context
    with mlflow.start_run():
        params = {
            "objective": "multi:softmax",
            "num_class": len(label_encoder.classes_),
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100
        }
        
        # Log the hyperparameters
        mlflow.log_params(params)

        print("Training XGBoost Router...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluation and Registry
        print("Evaluating model...")
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        # Log the performance metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save the actual model artifact into MLflow
        mlflow.xgboost.log_model(model, "xgboost_intent_model")
        
        print(f"Run completed. Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_model()