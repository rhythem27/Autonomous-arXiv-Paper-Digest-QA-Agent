"""Autonomous arXiv Paper Digest & QA Agent.

Powered by LangGraph, Gemini 2.5 Flash, Qdrant Local, and PyMuPDF.
"""

from src.config import Settings, settings
from src.logger import get_logger

__version__ = "0.1.0"
__all__ = ["Settings", "settings", "get_logger", "__version__"]
