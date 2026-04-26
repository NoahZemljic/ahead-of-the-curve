import logging
from datetime import datetime, timezone

from huggingface_hub import HfApi, ModelCard

logger = logging.getLogger(__name__)

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

def fetch_robotics_models(limit: int | None = None) -> list[dict]:
    """Fetch general-purpose robotics models from the Hugging Face Hub.

    Queries models with pipeline_tag='robotics', sorted by trending score
    descending so that the most relevant general-purpose models appear first.

    Args:
        limit: Maximum number of models to return. None fetches all.

    Returns:
        List of model dictionaries with expanded metadata.
    """
    api = HfApi()

    models = api.list_models(
        pipeline_tag="robotics",
        sort="trending_score",
        expand=EXPAND_FIELDS,
        limit=limit,
    )

    results = []
    for model in models:
        card_text = _fetch_card_text(model.id)
        results.append(
            {
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
                "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
            }
        )

    return results


def _fetch_card_text(model_id: str) -> str | None:
    """Fetch the model card README text for a given model."""
    try:
        card = ModelCard.load(model_id)
        return card.text if card.text else None
    except Exception:
        logger.warning("Failed to fetch model card for %s", model_id)
        return None

if __name__ == "__main__":
    models = fetch_robotics_models(limit=1)
    print(models[0]['card_text'])
