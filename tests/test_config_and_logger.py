"""Unit tests for configuration management and structured logging."""

import io
import logging
import os
import shutil
from pathlib import Path
import pytest

from src.config import Settings, settings
from src.logger import get_log_level, get_logger


class TestSettings:
    """Test suite for Settings configuration."""

    def test_default_settings_loaded(self):
        """Verify default settings values match expected project configuration."""
        assert settings.gemini_model == "gemini-2.5-flash"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.qdrant_path == "./qdrant_storage"
        assert settings.sqlite_db_path == "./agent_state.db"
        assert settings.pdf_cache_dir == "./pdf_cache"
        assert settings.arxiv_max_results == 5
        assert settings.top_k_chunks == 4
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 150
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_custom_settings_override(self, tmp_path):
        """Verify custom settings overrides and validation."""
        test_cache = str(tmp_path / "custom_pdf_cache")
        test_qdrant = str(tmp_path / "custom_qdrant")
        test_db = str(tmp_path / "sub" / "custom_state.db")

        custom_settings = Settings(
            gemini_model="gemini-1.5-pro",
            pdf_cache_dir=test_cache,
            qdrant_path=test_qdrant,
            sqlite_db_path=test_db,
            arxiv_max_results=10,
            log_level="DEBUG",
        )

        assert custom_settings.gemini_model == "gemini-1.5-pro"
        assert custom_settings.arxiv_max_results == 10
        assert custom_settings.log_level == "DEBUG"
        assert Path(test_cache).exists()
        assert Path(test_qdrant).exists()
        assert Path(test_db).parent.exists()

    def test_ensure_directories(self, tmp_path):
        """Verify ensure_directories creates required directories."""
        cache_dir = tmp_path / "test_cache"
        qdrant_dir = tmp_path / "test_qdrant"
        db_file = tmp_path / "nested" / "test.db"

        custom_settings = Settings(
            pdf_cache_dir=str(cache_dir),
            qdrant_path=str(qdrant_dir),
            sqlite_db_path=str(db_file),
        )

        assert cache_dir.is_dir()
        assert qdrant_dir.is_dir()
        assert db_file.parent.is_dir()


class TestLogger:
    """Test suite for structured logging utility."""

    def test_get_log_level_resolution(self):
        """Verify log level resolution from strings."""
        assert get_log_level("DEBUG") == logging.DEBUG
        assert get_log_level("info") == logging.INFO
        assert get_log_level("WARNING") == logging.WARNING
        assert get_log_level("error") == logging.ERROR
        assert get_log_level("critical") == logging.CRITICAL
        assert get_log_level("UNKNOWN_LEVEL") == logging.INFO

    def test_get_logger_instance(self):
        """Verify logger is created with proper handler and level."""
        logger_name = "test_logger_unique"
        logger = get_logger(logger_name)

        assert isinstance(logger, logging.Logger)
        assert logger.name == logger_name
        assert len(logger.handlers) == 1
        assert not logger.propagate

        # Verify idempotency (no duplicate handlers added)
        logger_again = get_logger(logger_name)
        assert logger_again is logger
        assert len(logger_again.handlers) == 1

    def test_logger_output_format(self, monkeypatch):
        """Verify log messages are properly formatted."""
        logger_name = "test_formatting_logger"
        logger = get_logger(logger_name)

        # Replace stream with StringIO
        stream = io.StringIO()
        logger.handlers[0].stream = stream

        test_message = "Sample log verification message"
        logger.info(test_message)

        output = stream.getvalue()
        assert "INFO" in output
        assert logger_name in output
        assert test_message in output
        assert "|" in output


def test_package_exports():
    """Verify package level exports from src."""
    import src

    assert hasattr(src, "settings")
    assert hasattr(src, "get_logger")
    assert hasattr(src, "Settings")
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"
