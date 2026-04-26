import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

EXPAND_FIELDS = [
    "cardData",
    "downloads",
    "downloadsAllTime",
    "createdAt",
    "lastModified",
    "likes",
    "trendingScore",
    "tags",
    "library_name",
]


def _model_to_dict(model, snapshot_date: str) -> dict:
    """Convert an HF ModelInfo object to a flat dict with card text."""
    return {
        "model_id": model.id,
        "author": model.author,
        "card_text": _fetch_card_text(model.id),
        "created_at": model.created_at,
        "last_modified": model.last_modified,
        "downloads_30d": model.downloads,
        "downloads_all_time": model.downloads_all_time,
        "likes": model.likes,
        "trending_score": model.trending_score,
        "tags": model.tags,
        "snapshot_date": snapshot_date,
    }


def fetch_robotics_models(limit: int | None = None) -> list[dict]:
    """Fetch general-purpose robotics models from the Hugging Face Hub.

    Queries models with pipeline_tag='robotics', sorted by trending score
    descending so that the most relevant general-purpose models appear first.

    Args:
        limit: Maximum number of models to return. None fetches all.

    Returns:
        List of model dictionaries with expanded metadata.
    """
    api = HfApi(token=HF_TOKEN)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()

    models = api.list_models(
        pipeline_tag="robotics",
        sort="trending_score",
        expand=EXPAND_FIELDS,
        limit=limit,
    )

    return [_model_to_dict(m, snapshot_date) for m in models]


def fetch_backfill_models(days: int = 90) -> list[dict]:
    """Fetch all robotics models created within the last `days` days.

    Args:
        days: Number of days to look back from today.

    Returns:
        List of model dictionaries, sorted by creation date (newest first).
    """
    api = HfApi(token=HF_TOKEN)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()

    models = api.list_models(
        pipeline_tag="robotics",
        sort="downloads",
        expand=EXPAND_FIELDS,
        limit=1000
    )

    results = []
    for model in models:
        if model.created_at is None or model.created_at < cutoff:
            continue
        results.append(_model_to_dict(model, snapshot_date))

    logger.info("Fetched %d models created in the last %d days", len(results), days)
    return results


def _fetch_card_text(model_id: str) -> str | None:
    """Fetch the model card README text for a given model."""
    try:
        card = ModelCard.load(model_id, token=HF_TOKEN)
        return card.text if card.text else None
    except Exception:
        logger.warning("Failed to fetch model card for %s", model_id)
        return None


if __name__ == "__main__":
    models = fetch_robotics_models(limit=1)
    print(models[0]["card_text"])
