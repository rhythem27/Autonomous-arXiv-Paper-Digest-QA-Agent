"""Configuration settings for Autonomous arXiv Paper Digest & QA Agent.

Loads configuration from environment variables and .env file with strongly-typed
defaults using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration (Google Gemini Free Tier via Google AI Studio)
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key from AI Studio",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier",
    )

    # Embeddings Configuration (BAAI/bge-small-en-v1.5 - Local / FastEmbed)
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedding model identifier",
    )

    # Vector Database Configuration (Qdrant Local)
    qdrant_path: str = Field(
        default="./qdrant_storage",
        description="Filesystem path for embedded local Qdrant on-disk storage",
    )
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Optional URL if running Qdrant server (e.g., http://localhost:6333)",
    )
    qdrant_collection_prefix: str = Field(
        default="arxiv",
        description="Prefix for paper collections in Qdrant",
    )

    # State Persistence (SQLite for session & checkpointing)
    sqlite_db_path: str = Field(
        default="./agent_state.db",
        description="Path to local SQLite database for session state",
    )

    # Storage & Cache Configuration
    pdf_cache_dir: str = Field(
        default="./pdf_cache",
        validation_alias=AliasChoices("pdf_cache_dir", "cache_dir"),
        description="Directory for caching downloaded arXiv PDFs",
    )

    @property
    def cache_dir(self) -> str:
        """Alias for pdf_cache_dir for notebook and CLI compatibility."""
        return self.pdf_cache_dir


    # Search & Retrieval Parameters
    arxiv_max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of search results to fetch from arXiv",
    )
    top_k_chunks: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Top-k chunks to retrieve for RAG",
    )
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Character count or token approximation per text chunk",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks",
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    def model_post_init(self, __context) -> None:
        """Ensure required local storage directories exist upon initialization."""
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create storage and cache directories if they do not already exist."""
        # PDF Cache directory
        if self.pdf_cache_dir:
            Path(self.pdf_cache_dir).mkdir(parents=True, exist_ok=True)

        # Embedded Qdrant storage directory (only when not using external URL)
        if not self.qdrant_url and self.qdrant_path:
            Path(self.qdrant_path).mkdir(parents=True, exist_ok=True)

        # Parent directory for SQLite DB if specified
        if self.sqlite_db_path:
            db_dir = Path(self.sqlite_db_path).parent
            if db_dir and not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
