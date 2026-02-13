"""Application configuration loaded from environment variables.

Uses pydantic-settings for validation and type-safe loading from .env.
Variables are read from .env (e.g., ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Required: ANTHROPIC_API_KEY, OPENAI_API_KEY
    All other fields have defaults and can be overridden via .env
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required API keys
    anthropic_api_key: str
    openai_api_key: str

    # LLM configuration
    llm_model_name: str = "claude-sonnet-4-20250514"
    embedding_model_name: str = "text-embedding-3-small"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

    # Chunking configuration
    chunk_size: int = 512
    chunk_overlap: int = 50

    # RAG retrieval configuration
    top_k_results: int = 5

    # Storage
    vector_db_path: str = "./data/vectordb"

    # Application metadata
    app_name: str = "Career Intelligence Assistant"
    log_level: str = "INFO"

    @field_validator("anthropic_api_key", "openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Ensure API keys are non-empty and not placeholder values."""
        if not v or v.strip().startswith("your_"):
            raise ValueError("API key must be set to a valid value (not placeholder)")
        return v


settings = Settings()
