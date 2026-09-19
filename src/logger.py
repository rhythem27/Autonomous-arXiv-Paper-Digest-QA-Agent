"""Centralized logging configuration for Autonomous arXiv Paper Digest & QA Agent.

Provides standardized, structured logging across all agent components, tools,
and pipeline stages.
"""

import logging
import sys
import warnings
from typing import Optional

from src.config import settings

# Suppress noisy library warnings
warnings.filterwarnings("ignore")

# Default structured log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log level mapping
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_CLI_SILENT: bool = False


class NoAfcWarningFilter(logging.Filter):
    """Filter out noisy Google GenAI Automatic Function Calling (AFC) warning messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if "automatic function calling" in msg or "(AFC)" in msg:
            return False
        return True


_afc_filter = NoAfcWarningFilter()

# Suppress Google GenAI internal AFC warning flags
try:
    import google.genai.models

    google.genai.models.Models._logged_afc_warning = True
    google.genai.models.AsyncModels._logged_afc_warning = True
except Exception:
    pass

# Attach AFC filter to root and google_genai loggers
logging.getLogger().addFilter(_afc_filter)
for _gname in ("google_genai", "google_genai.models", "google"):
    _gl = logging.getLogger(_gname)
    _gl.addFilter(_afc_filter)


def set_cli_silent(silent: bool = True) -> None:
    """Enable or disable console log suppression across all loggers for clean CLI presentation.

    Args:
        silent: If True, sets all internal and third-party loggers to ERROR level.
    """
    global _CLI_SILENT
    _CLI_SILENT = silent
    target_level = logging.ERROR if silent else get_log_level()

    # Set root logger level
    logging.getLogger().setLevel(target_level)

    # Set all registered loggers and their handlers
    for name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if isinstance(logger_obj, logging.Logger):
            if name.startswith("src") or name in (
                "agent_graph",
                "agent_nodes",
                "gemini_client",
                "cli",
                "arxiv_agent",
                "google_genai",
                "google_genai.models",
                "google",
            ):
                logger_obj.setLevel(target_level)
                for handler in logger_obj.handlers:
                    handler.setLevel(target_level)


def get_log_level(level_name: Optional[str] = None) -> int:
    """Resolve a string log level to its Python logging constant.

    Args:
        level_name: Case-insensitive level name (e.g., 'DEBUG', 'INFO').
                    If None, uses `settings.log_level`.

    Returns:
        int: Python logging level constant.
    """
    name = (level_name or settings.log_level).strip().upper()
    return _LOG_LEVELS.get(name, logging.INFO)


def get_logger(name: str = "arxiv_agent") -> logging.Logger:
    """Retrieve or create a configured logger instance.

    Ensures that stream handlers are attached without duplicate log propagation,
    adhering to the configured log level and format.

    Args:
        name: Name of the logger (typically __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    level = logging.ERROR if _CLI_SILENT else get_log_level()
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        handler.addFilter(_afc_filter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.addFilter(_afc_filter)

    # Prevent double-logging when root logger also has handlers
    logger.propagate = False

    return logger

