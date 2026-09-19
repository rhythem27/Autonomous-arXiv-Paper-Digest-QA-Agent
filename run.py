#!/usr/bin/env python
"""Autonomous arXiv Paper Digest & QA Agent - CLI Entrypoint.

Usage:
  python run.py
  python run.py --query "1706.03762"
  python run.py --query "attention mechanism" --question "How does self-attention work?" --no-interactive
"""

import sys
import logging
import warnings

# Suppress noisy library warnings (e.g. langchain sampling warnings)
warnings.filterwarnings("ignore")

# Ensure UTF-8 output encoding across Windows consoles to support math symbols and Unicode
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silence google-genai automatic function calling warning
try:
    import google.genai.models
    google.genai.models.Models._logged_afc_warning = True
    google.genai.models.AsyncModels._logged_afc_warning = True
except Exception:
    pass

from src.logger import set_cli_silent
set_cli_silent(True)

from src.cli.main import app


if __name__ == "__main__":
    app()
