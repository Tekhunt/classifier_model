"""
config.py - Application Configuration (No Pydantic dependency)
"""

import os
from typing import Optional, List, Tuple

class Settings:
    """Simple configuration class using environment variables or defaults."""

    # API Settings
    app_name: str = os.getenv("APP_NAME", "Text Classification Midterm Project")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    api_prefix: str = os.getenv("API_PREFIX", "/api")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Server Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 6000))
    reload: bool = os.getenv("RELOAD", "True").lower() == "true"

    # CORS Settings
    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"
    cors_allow_methods: List[str] = os.getenv("CORS_ALLOW_METHODS", "*").split(",")
    cors_allow_headers: List[str] = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")

    # Model Settings
    model_path: str = os.getenv("MODEL_PATH", "models/classifier.pkl")
    vectorizer_path: str = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")
    max_features: int = int(os.getenv("MAX_FEATURES", 5000))
    ngram_range: Tuple[int, int] = (1, 2)
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

    # Data Settings
    sample_data_path: str = os.getenv("SAMPLE_DATA_PATH", "data/sample_data.json")
    dataset_path: Optional[str] = os.getenv("DATASET_PATH", "data/user_comments.tsv")

    # Pagination Settings
    default_page_size: int = int(os.getenv("DEFAULT_PAGE_SIZE", 10))
    max_page_size: int = int(os.getenv("MAX_PAGE_SIZE", 100))

    # Training Settings
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    random_state: int = int(os.getenv("RANDOM_STATE", 42))
    max_iter: int = int(os.getenv("MAX_ITER", 1000))

    # Database Settings
    database_url: Optional[str] = os.getenv("DATABASE_URL", "sqlite:///./sentiment_analysis.db")
    use_sqlite: bool = os.getenv("USE_SQLITE", "True").lower() == "true"

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")

# Create global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
