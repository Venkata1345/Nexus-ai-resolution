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
    models_dir: Path = PROJECT_ROOT / "models"

    # -------- MLflow --------
    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow_experiment_name: str = "nexus_intent_classification"
    mlflow_model_artifact_name: str = "xgboost_intent_model"

    # -------- Training hyperparameters --------
    tfidf_max_features: int = 5000
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    train_test_split_ratio: float = 0.2
    random_seed: int = 42

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the singleton Settings instance."""
    return Settings()  # type: ignore[call-arg]


# Import-friendly alias. Prefer `from src.config import settings` in app code.
settings = get_settings()
