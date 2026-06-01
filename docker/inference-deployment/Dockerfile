FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group inference --no-install-project --no-cache

COPY src/inference_pipeline/__init__.py      ./inference_pipeline/
COPY src/inference_pipeline/main.py          ./inference_pipeline/
COPY src/inference_pipeline/preprocessing.py ./inference_pipeline/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8080

CMD ["uvicorn", "inference_pipeline.main:app", "--host", "0.0.0.0", "--port", "8080"]
