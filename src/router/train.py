import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)


def train_model():
    print("Loading data...")
    df = pd.read_csv(settings.raw_data_path)

    X = df["instruction"]
    y = df["intent"]

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=settings.tfidf_max_features)
    X_vec = vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, settings.vectorizer_path)
    joblib.dump(label_encoder, settings.label_encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y_enc,
        test_size=settings.test_fraction,
        random_state=settings.random_seed,
    )

    with mlflow.start_run():
        params = {
            "objective": "multi:softmax",
            "num_class": len(label_encoder.classes_),
            "max_depth": settings.xgb_max_depth,
            "learning_rate": settings.xgb_learning_rate,
            "n_estimators": settings.xgb_n_estimators,
        }

        mlflow.log_params(params)

        print("Training XGBoost Router...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        print("Evaluating model...")
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.xgboost.log_model(model, settings.mlflow_model_artifact_name)

        print(f"Run completed. Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    train_model()
