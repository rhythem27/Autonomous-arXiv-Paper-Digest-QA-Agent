"""Centralized logging configuration for Autonomous arXiv Paper Digest & QA Agent.

Provides standardized, structured logging across all agent components, tools,
and pipeline stages.
"""

import logging
import sys
from typing import Optional

from src.config import settings

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
    level = get_log_level()
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent double-logging when root logger also has handlers
    logger.propagate = False

    return logger
