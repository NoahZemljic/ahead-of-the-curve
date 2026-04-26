from datetime import datetime, timezone
from huggingface_hub import HfApi

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
        results.append(
            {
                "model_id": model.id,
                "author": model.author,
                "card_data": model.card_data,
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
