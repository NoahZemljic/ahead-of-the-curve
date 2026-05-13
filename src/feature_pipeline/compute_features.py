import logging
import re
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FeatureComputer:
    """Compute semantic, temporal, velocity, and metadata features for models."""

    def __init__(self):
        """Initialize topic reference embeddings and the sentence-transformer encoder."""
        self.TOPIC_REFERENCES = {
            "robotics": (
                "vla, vision language action model, general-purpose robot manipulation, general "
                "embodied ai agent, robotic foundation model for dexterous grasping, "
                "navigation, and multi-task control in real-world environments," "flow matching, multimodal, robot actions, lerobot"
            ),
            "slm": (
                "small language model optimized for edge deployment and on-device inference, "
                "lightweight transformer for mobile and embedded systems, efficient llm "
                "with low latency and minimal memory footprint for real-time applications"
            ),
            "multimodal_reasoning": (
                "multimodal reasoning model that jointly processes text, images, and video "
                "for complex visual question answering, chain-of-thought reasoning across "
                "modalities, and structured problem solving with visual understanding,"
                "multimodal"
            ),
        }
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.topic_embeddings = {
            topic: self.encoder.encode(text, normalize_embeddings=True)
            for topic, text in self.TOPIC_REFERENCES.items()
        }

    def clean_card_text(self, text: str | None) -> str | None:
        """Strip HTML tags, markdown images/links, URLs, and table rows from card text."""
        if not text:
            return None

        # HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Markdown images
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        # Markdown links
        text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Bare URLs
        text = re.sub(r"https?://\S+", "", text)
        # Markdown table rows
        text = re.sub(r"\|[^\n]+\|", "", text)
        # Markdown formatting characters
        text = re.sub(r"[#*_>{}\[\]]+", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text if text else None

    def set_timezone_utc(self, dt: datetime) -> datetime:
        """Attach UTC timezone if the datetime is naive."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def find_downloads_at_age(
        self,
        prior_snapshots: list[dict],
        created_at: datetime,
        min_hours: float,
        max_hours: float,
    ) -> int | None:
        """Find downloads_30d from a snapshot taken within [min_hours, max_hours] after model creation."""
        for snapshot in prior_snapshots:
            snap_time = self.set_timezone_utc(datetime.fromisoformat(snapshot["snapshot_date"]))
            hours_since_creation = (snap_time - created_at).total_seconds() / 3600

            if min_hours <= hours_since_creation <= max_hours:
                return snapshot.get("downloads_30d")

        return None

    def compute_semantic_relevance(self, card_embedding: np.ndarray | None) -> dict[str, float]:
        """Compute topic relevance scores and the best-matching frontier topic."""
        if card_embedding is None:
            return {
                "relevance_robotics": 0.0,
                "relevance_slm": 0.0,
                "relevance_multimodal": 0.0,
                "best_topic": "unknown",
                "best_topic_score": 0.0,
            }

        scores = {
            topic: float(np.dot(card_embedding, topic_emb))
            for topic, topic_emb in self.topic_embeddings.items()
        }

        best_topic = max(scores, key=scores.get)

        return {
            "relevance_robotics": scores["robotics"],
            "relevance_slm": scores["slm"],
            "relevance_multimodal": scores["multimodal_reasoning"],
            "best_topic": best_topic,
            "best_topic_score": scores[best_topic],
        }

    def compute_age_hours(
        self, created_at: datetime | None, snapshot_date: str
    ) -> float | None:
        """Compute the age of a model in hours at the time of the snapshot."""
        if created_at is None:
            return None

        now = self.set_timezone_utc(datetime.fromisoformat(snapshot_date))
        created_at = self.set_timezone_utc(created_at)
        return (now - created_at).total_seconds() / 3600

    def compute_download_velocity(
        self,
        created_at: datetime | None,
        prior_snapshots: list[dict] | None = None,
    ) -> dict[str, int | None]:
        """Compute downloads in the first 24h and 72h windows since model creation.

        Looks for a snapshot taken ~24h / ~72h after creation (±4h tolerance)
        and returns its downloads_30d as the velocity for that window.
        """
        if not created_at or not prior_snapshots:
            return {"download_velocity_24h": None, "download_velocity_72h": None}

        created_at = self.set_timezone_utc(created_at)

        return {
            "download_velocity_24h": self.find_downloads_at_age(
                prior_snapshots, created_at, min_hours=20, max_hours=28
            ),
            "download_velocity_72h": self.find_downloads_at_age(
                prior_snapshots, created_at, min_hours=68, max_hours=76
            ),
        }

    def compute_metadata_features(self, model: dict) -> dict:
        """Extract raw metadata features from a model dict."""
        tags = model.get("tags") or []

        return {
            "likes": model.get("likes", 0),
            "trending_score": model.get("trending_score", 0),
            "downloads_30d": model.get("downloads_30d", 0),
            "downloads_all_time": model.get("downloads_all_time", 0),
            "tag_count": len(tags),
            "has_paper_tag": any(tag.startswith("arxiv:") for tag in tags),
        }

    def compute_features_batch(
        self,
        models: list[dict],
        prior_snapshots: dict[str, list[dict]] | None = None,
        batch_size: int = 64,
    ) -> list[dict]:
        """Compute feature rows for multiple models with batched text encoding."""
        # Clean all card texts
        cleaned_texts = [self.clean_card_text(m.get("card_text")) for m in models]

        # Batch-encode texts
        texts_to_encode = [t.lower() for t in cleaned_texts if t]
        encodable_indices = [i for i, t in enumerate(cleaned_texts) if t]

        embeddings = {}
        if texts_to_encode:
            encoded = self.encoder.encode(
                texts_to_encode, normalize_embeddings=True, batch_size=batch_size
            )
            for idx, emb in zip(encodable_indices, encoded):
                embeddings[idx] = emb

        # Assemble features using precomputed embeddings
        results = []
        for i, model in enumerate(models):
            card_embedding = embeddings.get(i)
            semantic_relevance = self.compute_semantic_relevance(card_embedding)
            age_hours = self.compute_age_hours(model["created_at"], model["snapshot_date"])
            model_snapshots = prior_snapshots.get(model["model_id"]) if prior_snapshots else None
            velocity = self.compute_download_velocity(model["created_at"], model_snapshots)
            metadata = self.compute_metadata_features(model)

            results.append({
                "model_id": model["model_id"],
                "created_at": model["created_at"],
                "snapshot_date": model["snapshot_date"],
                **semantic_relevance,
                "age_hours": age_hours,
                **velocity,
                **metadata,
            })

        has_24h = sum(1 for r in results if r["download_velocity_24h"] is not None)
        has_72h = sum(1 for r in results if r["download_velocity_72h"] is not None)
        logger.info(
            f"Download velocity computed for {has_24h} / {len(results)} models (24h) "
            f"and {has_72h} / {len(results)} models (72h)"
        )

        return results
