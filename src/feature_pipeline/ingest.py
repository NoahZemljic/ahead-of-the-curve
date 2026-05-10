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

BACKFILL_TOPICS = [
    {"pipeline_tag": "robotics"},
    {"pipeline_tag": "image-text-to-text"},
    {"pipeline_tag": "text-generation", "limit": 5000},
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


def fetch_models(since_days: int = 1) -> list[dict]:
    """Fetch the first 3000 created models from the Hugging Face Hub.

    Used by the daily pipeline to discover new models across all categories.
    Models are sorted by creation date and only those created within
    the given window are returned.
    """
    api = HfApi(token=HF_TOKEN)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)

    models = []
    for model in api.list_models(
        sort="created_at",
        limit=3000,
        expand=EXPAND_FIELDS,
    ):
        if model.last_modified and model.last_modified < cutoff_date:
            break
        models.append(model)

    card_texts = _fetch_card_texts_parallel([m.id for m in models])
    results = [_model_to_dict(m, snapshot_date, card_texts[m.id]) for m in models]

    logger.info(f"Fetched {len(results)} models modified in the last {since_days} day(s)")
    return results


def fetch_models_backfill(since_days: int = 90) -> list[dict]:
    """Fetch models from frontier topics (robotics, multimodal reasoning, SLMs) for historical backfill.

    Queries each top separately, combines results, deduplicates, and filters to models
    created within the given time window.
    """
    api = HfApi(token=HF_TOKEN)
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)

    seen_models = set()
    all_models = []

    for topic in BACKFILL_TOPICS:
        topic_models = list(api.list_models(
            pipeline_tag=topic["pipeline_tag"],
            sort="trending_score",
            expand=EXPAND_FIELDS,
            limit=topic.get("limit"),
        ))

        for m in topic_models:
            if m.id not in seen_models:
                seen_models.add(m.id)
                all_models.append(m)

        logger.info(f"Fetched {len(topic_models)} models for topic '{topic['pipeline_tag']}'")

    all_models = [m for m in all_models if m.created_at and m.created_at >= cutoff_date]

    card_texts = _fetch_card_texts_parallel([m.id for m in all_models])
    results = [_model_to_dict(m, snapshot_date, card_texts[m.id]) for m in all_models]

    logger.info(f"Backfill: {len(results)} models across {len(BACKFILL_TOPICS)} niches (last {since_days} days)")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = fetch_models(since_days=1)
    print(f"Daily: {len(models)} models fetched.")
