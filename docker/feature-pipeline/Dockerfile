FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN uv sync --frozen --no-dev --group feature --no-install-project --no-cache

COPY ./src/feature_pipeline/ .

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python"]
CMD ["pipeline.py"]
