FROM python:3.13-slim

WORKDIR /app

# Build dependencies for C extensions (lxml, fastembed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install all dependencies
RUN pip install --no-cache-dir -e "."

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Pre-download BM42 sparse model at build time to avoid cold-start delay
# Must run as appuser so the cache lands in /home/appuser/.cache/fastembed/
RUN python -c "\
from fastembed import SparseTextEmbedding; \
SparseTextEmbedding('Qdrant/bm42-all-minilm-l6-v2-attentions')"

EXPOSE 8080

CMD ["uvicorn", "stripe_rag.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
