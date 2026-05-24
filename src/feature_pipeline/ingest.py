import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard

load_dotenv()

logger = logging.getLogger(__name__)
T = TypeVar("T")


class HFIngestor:
    """Fetch model metadata and card texts from the Hugging Face Hub."""

    def __init__(self):
        """Initialize Hugging Face API settings, fetch limits, and retry policy."""
        self.hf_token = os.getenv("HF_TOKEN")
        self.EXPAND_FIELDS = [
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
        self.BACKFILL_TOPICS = [
            {"pipeline_tag": "robotics"},
            {"pipeline_tag": "image-text-to-text"},
            {"pipeline_tag": "text-generation", "limit": 5000},
        ]
        self.DAILY_FETCH_LIMIT = 4000
        self.BACKOFF_SCHEDULE = (1, 5, 10, 30, 60, 300)

    def fetch_with_backoff(
        self,
        fetch_fn: Callable[[], T],
        description: str,
        max_retries: int | None = None,
    ) -> T | None:
        """Run a Hugging Face fetch (model info / card) with retry delays for 429 rate limits."""
        if max_retries is None:
            max_retries = len(self.BACKOFF_SCHEDULE)

        for attempt in range(max_retries):
            try:
                return fetch_fn()
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = self.BACKOFF_SCHEDULE[attempt]
                    logger.warning(f"Rate limited fetching {description}, retrying in {wait}s")
                    time.sleep(wait)
                    continue

                logger.warning(f"Failed to fetch {description}: {e}")
                return None

    def model_to_dict(self, model, snapshot_date: str, card_text: str | None = None) -> dict:
        """Convert an HF ModelInfo object to a flat dict with card text."""
        return {
            "model_id": model.id,
            "author": model.author,
            "card_text": card_text,
            "created_at": model.created_at,
            "last_modified": model.last_modified,
            "downloads_30d": model.downloads or 0,
            "downloads_all_time": model.downloads_all_time or 0,
            "likes": model.likes or 0,
            "trending_score": model.trending_score or 0,
            "tags": model.tags,
            "snapshot_date": snapshot_date,
        }

    def fetch_models(self, since_days: int = 1) -> list[dict]:
        """Fetch the first DAILY_FETCH_LIMIT created models from the Hugging Face Hub.

        Used by the daily pipeline to discover new models across all categories.
        Models are sorted by creation date and only those created within
        the given window are returned.
        """
        api = HfApi(token=self.hf_token)
        snapshot_date = datetime.now(timezone.utc).isoformat()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)

        models = []
        for model in api.list_models(
            sort="created_at",
            limit=self.DAILY_FETCH_LIMIT,
            expand=self.EXPAND_FIELDS,
        ):
            if model.last_modified and model.last_modified < cutoff_date:
                break
            models.append(model)

        card_texts = self.fetch_card_texts_parallel([model.id for model in models])
        results = [self.model_to_dict(m, snapshot_date, card_texts[m.id]) for m in models]

        logger.info(f"Fetched {len(results)} models modified in the last {since_days} day(s)")
        return results

    def fetch_models_backfill(self, since_days: int = 90) -> list[dict]:
        """Fetch models from frontier topics (robotics, multimodal reasoning, SLMs) for historical backfill.

        Queries each top separately, combines results, deduplicates, and filters to models
        created within the given time window.
        """
        api = HfApi(token=self.hf_token)
        snapshot_date = datetime.now(timezone.utc).isoformat()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)

        seen_models = set()
        all_models = []

        for topic in self.BACKFILL_TOPICS:
            topic_models = list(api.list_models(
                pipeline_tag=topic["pipeline_tag"],
                sort="trending_score",
                expand=self.EXPAND_FIELDS,
                limit=topic.get("limit"),
            ))

            for m in topic_models:
                if m.id not in seen_models:
                    seen_models.add(m.id)
                    all_models.append(m)

            logger.info(f"Fetched {len(topic_models)} models for topic '{topic['pipeline_tag']}'")

        all_models = [m for m in all_models if m.created_at and m.created_at >= cutoff_date]

        card_texts = self.fetch_card_texts_parallel([m.id for m in all_models])
        results = [self.model_to_dict(m, snapshot_date, card_texts[m.id]) for m in all_models]

        logger.info(f"Backfill: {len(results)} models across {len(self.BACKFILL_TOPICS)} topics (last {since_days} days)")
        return results

    def fetch_models_by_id(self, model_ids: list[str]) -> list[dict]:
        """Fetch current state of specific models by their IDs.
        """
        api = HfApi(token=self.hf_token)
        snapshot_date = datetime.now(timezone.utc).isoformat()

        models = []
        for model_id in model_ids:
            model = self.fetch_with_backoff(
                lambda model_id=model_id: api.model_info(model_id, expand=self.EXPAND_FIELDS),
                f"model info for {model_id}",
            )
            if model:
                models.append(model)

        card_texts = self.fetch_card_texts_parallel([m.id for m in models])
        results = [self.model_to_dict(m, snapshot_date, card_texts[m.id]) for m in models]

        logger.info(f"Re-fetched {len(results)} models by ID.")
        return results

    def fetch_card_texts_parallel(
        self, model_ids: list[str], max_workers: int = 4, request_delay: float = 0.25
    ) -> dict[str, str | None]:
        """Fetch model card texts in parallel using a thread pool.

        Submissions are staggered by request_delay seconds to avoid bursting
        all workers simultaneously and triggering HF Hub rate limits.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for model_id in model_ids:
                futures[executor.submit(self.fetch_card_text, model_id)] = model_id
                time.sleep(request_delay)
            for future in as_completed(futures):
                model_id = futures[future]
                results[model_id] = future.result()
        return results

    def fetch_recent_models(self, hours: int = 70) -> list[dict]:
        """Fetch models created within the last `hours` hours, newest first.

        Used by the inference pipeline to discover models uploaded since the
        previous hourly run without fetching the entire daily batch.
        """
        api = HfApi(token=self.hf_token)
        snapshot_date = datetime.now(timezone.utc).isoformat()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        models = []
        for model in api.list_models(
            sort="created_at",
            expand=self.EXPAND_FIELDS,
        ):
            if not model.created_at or model.created_at < cutoff:
                break
            models.append(model)

        card_texts = self.fetch_card_texts_parallel([m.id for m in models])
        results = [self.model_to_dict(m, snapshot_date, card_texts[m.id]) for m in models]
        logger.info(f"Fetched {len(results)} models created in the last {hours} hours")
        return results

    def fetch_card_text(self, model_id: str, max_retries: int | None = None) -> str | None:
        """Fetch the model card README text for a given model, with retry on 429."""
        card = self.fetch_with_backoff(
            lambda: ModelCard.load(model_id, token=self.hf_token),
            f"model card for {model_id}",
            max_retries=max_retries,
        )
        return card.text if card and card.text else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingestor = HFIngestor()
    models = ingestor.fetch_models(since_days=1)
    print(f"Daily: {len(models)} models fetched.")
