FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binaries
COPY --from=uv /uv /uvx /bin/

# Copy dependencies first for caching
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy download script and pre-download models
COPY download_models.py .
RUN .venv/bin/python download_models.py

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"

EXPOSE 8008

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
