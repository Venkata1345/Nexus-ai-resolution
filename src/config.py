"""Centralized, typed configuration for Nexus.

Every tunable — file paths, MLflow URIs, LLM parameters, secrets — lives here.
Import the singleton `settings` anywhere in the codebase; override any value by
setting an environment variable of the same (upper-cased) name, or by putting
it in a `.env` file at the project root.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the `src/` directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration for Nexus, loaded from env vars / .env with defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- Paths --------
    project_root: Path = PROJECT_ROOT
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "bitext_support_data.csv"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"

    # -------- MLflow --------
    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow_experiment_name: str = "nexus_intent_classification"
    # One artifact name per pipeline — easier to disambiguate in MLflow UI.
    mlflow_model_artifact_name: str = "xgboost_intent_model"  # used by TF-IDF pipeline
    mlflow_embedding_model_artifact_name: str = "xgboost_embedding_intent_model"
    # Model Registry: a single named entry whose @production alias is what
    # the agent loads at runtime. Decoupled from any specific run id.
    mlflow_registered_model_name: str = "nexus_intent_classifier"
    mlflow_production_alias: str = "production"

    # -------- Embedding model --------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # -------- RAG knowledge base (Chroma) --------
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"
    kb_collection_name: str = "nexus_support_faqs"
    kb_top_k: int = 3

    # -------- Routing --------
    # If the XGBoost confidence for the top intent is below this, the graph
    # escalates to a human instead of dispatching to a worker.
    intent_confidence_threshold: float = 0.3

    # -------- Hyperparameter tuning --------
    optuna_n_trials: int = 20

    # -------- Data split --------
    # 3-way stratified split. Train gets what's left after val + test.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42

    # -------- Training hyperparameters (baseline defaults; Optuna tunes these) --------
    tfidf_max_features: int = 5000
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100

    # -------- LLM --------
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    llm_temperature: float = 0.2
    # Required at runtime — pydantic raises a clear error if it's missing.
    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")

    # -------- Derived paths (computed from the values above) --------
    @property
    def vectorizer_path(self) -> Path:
        return self.models_dir / "tfidf_vectorizer.pkl"

    @property
    def label_encoder_path(self) -> Path:
        return self.models_dir / "label_encoder.pkl"

    @property
    def train_csv_path(self) -> Path:
        return self.processed_data_dir / "train.csv"

    @property
    def val_csv_path(self) -> Path:
        return self.processed_data_dir / "val.csv"

    @property
    def test_csv_path(self) -> Path:
        return self.processed_data_dir / "test.csv"

    @property
    def train_embeddings_path(self) -> Path:
        return self.processed_data_dir / "train_embeddings.npy"

    @property
    def val_embeddings_path(self) -> Path:
        return self.processed_data_dir / "val_embeddings.npy"

    @property
    def test_embeddings_path(self) -> Path:
        return self.processed_data_dir / "test_embeddings.npy"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]


# Import-friendly alias. Prefer `from src.config import settings` in app code.
settings = get_settings()
