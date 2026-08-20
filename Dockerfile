# =============================================================================
# Stage 1: base — Python 3.11 slim + all dependencies
# =============================================================================
FROM python:3.11-slim AS base

# System libraries needed by pdfplumber / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (avoid pulling in CUDA libs)
# Remove this layer and install nvidia/cuda torch for GPU support.
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY migrations/ ./migrations/

# NOTE: The embedding model (bge-large-en-v1.5 ~1.2 GB) is NOT baked into the
# image. It downloads on first container start and is cached in the hf_cache
# Docker volume. This keeps rebuilds fast (<30s after first build).
# To pre-bake for air-gapped deployments, uncomment:
# ARG EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# =============================================================================
# Stage 2: app — FastAPI server
# =============================================================================
FROM base AS app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Stage 3: worker — arq background worker
# =============================================================================
FROM base AS worker

CMD ["python", "-m", "arq", "app.worker.worker_settings.WorkerSettings"]
