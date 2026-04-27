import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _model_to_dict(model, snapshot_date: str, card_text: str | None = None) -> dict:
    """Convert an HF ModelInfo object to a flat dict with card text."""
    return {
        "model_id": model.id,
        "author": model.author,
        "card_text": card_text,
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

    models = list(api.list_models(
        pipeline_tag="robotics",
        sort="trending_score",
        expand=EXPAND_FIELDS,
        limit=limit,
    ))

    card_texts = _fetch_card_texts_parallel([m.id for m in models])
    return [_model_to_dict(m, snapshot_date, card_texts[m.id]) for m in models]


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
        limit=1,
        expand=EXPAND_FIELDS,
    )

    filtered = [m for m in models if m.created_at is not None and m.created_at >= cutoff]

    card_texts = _fetch_card_texts_parallel([m.id for m in filtered])
    results = [_model_to_dict(m, snapshot_date, card_texts[m.id]) for m in filtered]

    logger.info(f"Fetched {len(results)} models created in the last {days} days")
    return results

def _fetch_card_texts_parallel(model_ids: list[str], max_workers: int = 4) -> dict[str, str | None]:
    """Fetch model card texts in parallel using a thread pool."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_card_text, model_id): model_id for model_id in model_ids}
        for future in as_completed(futures):
            mid = futures[future]
            results[mid] = future.result()
    return results


def _fetch_card_text(model_id: str, max_retries: int = 3) -> str | None:
    """Fetch the model card README text for a given model, with retry on 429."""
    for attempt in range(max_retries):
        try:
            card = ModelCard.load(model_id, token=HF_TOKEN)
            return card.text if card.text else None
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Rate limited fetching {model_id}, retrying in {wait}s")
                time.sleep(wait)
                continue
            logger.warning(f"Failed to fetch model card for {model_id}: {e}")
            return None


if __name__ == "__main__":
    models = fetch_robotics_models(limit=1)
    print(models[0]["card_text"])
