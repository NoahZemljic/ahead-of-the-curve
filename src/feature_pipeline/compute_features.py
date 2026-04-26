import re
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import fetch_robotics_models

TOPIC_REFERENCES = {
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

def clean_card_text(text: str | None) -> str | None:
    """Strip HTML tags, markdown images/links, URLs, and table rows from card text."""
    if not text:
        return None

    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Markdown links [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Markdown table rows
    text = re.sub(r"\|[^\n]+\|", "", text)
    # Markdown formatting chars
    text = re.sub(r"[#*_>{}\[\]]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None

def set_timezone_utc(dt: datetime) -> datetime:
    """Attach UTC timezone if the datetime is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def _find_downloads_at_age(
    prior_snapshots: list[dict],
    created_at: datetime,
    min_hours: float,
    max_hours: float,
) -> int | None:
    """Find downloads_30d from a snapshot taken within [min_hours, max_hours] after model creation."""
    for snapshot in prior_snapshots:
        snap_time = set_timezone_utc(datetime.fromisoformat(snapshot["snapshot_date"]))
        hours_since_creation = (snap_time - created_at).total_seconds() / 3600

        if min_hours <= hours_since_creation <= max_hours:
            return snapshot.get("downloads_30d")

    return None

class FeatureComputer:
    def __init__(self):
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._topic_embeddings = {
            topic: self._encoder.encode(text, normalize_embeddings=True)
            for topic, text in TOPIC_REFERENCES.items()
        }

    def compute_semantic_relevance(self, card_embedding: np.ndarray | None) -> dict[str, float]:
        """Compute cosine similarity of a card embedding against each topic vector.

        Topics: robotics, small language models (SLMs), multimodal reasoning.

        Returns a dict with one score per topic plus the best-matching topic name.
        """
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
            for topic, topic_emb in self._topic_embeddings.items()
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

        now = set_timezone_utc(datetime.fromisoformat(snapshot_date))
        created_at = set_timezone_utc(created_at)
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

        created_at = set_timezone_utc(created_at)

        return {
            "download_velocity_24h": _find_downloads_at_age(
                prior_snapshots, created_at, min_hours=20, max_hours=28
            ),
            "download_velocity_72h": _find_downloads_at_age(
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
            "has_paper_tag": any(t.startswith("arxiv:") for t in tags),
        }

    def compute_features(self, model: dict, prior_snapshots: list[dict] | None = None) -> dict:
        """Compute the full feature row for a single model.

        Args:
            model: Model dict as returned by ingest.fetch_robotics_models().
            prior_snapshots: Previous daily snapshots for velocity computation.

        Returns:
            Flat dict with all features, ready for the feature store.
        """
        cleaned_text = clean_card_text(model.get("card_text"))

        card_embedding = None
        if cleaned_text:
            card_embedding = self._encoder.encode(cleaned_text.lower(), normalize_embeddings=True)

        # Compute features
        semantic_relevance = self.compute_semantic_relevance(card_embedding)
        age_hours = self.compute_age_hours(model["created_at"], model["snapshot_date"])
        velocity = self.compute_download_velocity(model["created_at"], prior_snapshots)
        metadata = self.compute_metadata_features(model)

        return {
            "model_id": model["model_id"],
            "created_at": model["created_at"],
            "snapshot_date": model["snapshot_date"],
            **semantic_relevance,
            "age_hours": age_hours,
            **velocity,
            **metadata,
        }


if __name__ == "__main__":
    computer = FeatureComputer()
    models = fetch_robotics_models(limit=1)

    for m in models:
        features = computer.compute_features(m)
        print(f"\n{'=' * 60}")
        print(f"Model: {features['model_id']}")
        print(f"  Relevance  robotics={features['relevance_robotics']:.3f}  "
              f"slm={features['relevance_slm']:.3f}  "
              f"multimodal={features['relevance_multimodal']:.3f}")
        print(f"  Best topic: {features['best_topic']} ({features['best_topic_score']:.3f})")
        print(f"  Age: {features['age_hours']:.1f}h" if features['age_hours'] else "  Age: N/A")
        print(f"  Likes: {features['likes']}  Trending: {features['trending_score']}  "
              f"Tags: {features['tag_count']}  Paper: {features['has_paper_tag']}")
