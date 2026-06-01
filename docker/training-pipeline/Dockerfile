FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN uv sync --frozen --no-dev --group training --no-install-project --no-cache

COPY src/common ./common
COPY src/training_pipeline ./training_pipeline

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "-m"]
CMD ["training_pipeline.pipeline"]
