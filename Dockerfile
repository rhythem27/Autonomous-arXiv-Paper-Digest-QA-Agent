FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.3.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system runtime & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency specifications first for Docker cache optimization
COPY pyproject.toml poetry.lock* README.md /app/

# Install dependencies (without root package)
RUN poetry install --no-interaction --no-root

# Copy application source code
COPY src/ /app/src/
COPY run.py /app/run.py

# Install project package
RUN poetry install --no-interaction

# Create cache and persistent store directories
RUN mkdir -p /app/pdf_cache /app/qdrant_storage

# Default command runs the CLI entrypoint
ENTRYPOINT ["python", "run.py"]
