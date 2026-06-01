FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN uv sync --frozen --no-dev --group inference-worker --no-install-project --no-cache

COPY src/common ./common
COPY src/feature_pipeline/__init__.py         ./feature_pipeline/
COPY src/feature_pipeline/ingest.py           ./feature_pipeline/
COPY src/feature_pipeline/compute_features.py ./feature_pipeline/
COPY src/inference_pipeline/__init__.py       ./inference_pipeline/
COPY src/inference_pipeline/hopsworks_store.py ./inference_pipeline/
COPY src/inference_pipeline/preprocessing.py  ./inference_pipeline/
COPY src/inference_pipeline/pipeline.py       ./inference_pipeline/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

CMD ["python", "-m", "inference_pipeline.pipeline"]
