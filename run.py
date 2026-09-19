#!/usr/bin/env python
"""Autonomous arXiv Paper Digest & QA Agent - CLI Entrypoint.

Usage:
  python run.py
  python run.py --query "1706.03762"
  python run.py --query "attention mechanism" --question "How does self-attention work?" --no-interactive
"""

import sys
from src.cli.main import app

if __name__ == "__main__":
    app()
