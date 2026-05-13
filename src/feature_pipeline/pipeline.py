import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from compute_features import FeatureComputer
from compute_labels import Labeller
from hopsworks_store import HopsworksStore
from ingest import HFIngestor

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Daily feature pipeline: fetch, score, filter, label, push to Hopsworks.

    Runs once per day at 00:00 UTC. Fetches all recently modified models from
    the Hub (regardless of tag), computes features including semantic relevance
    against frontier topics, filters to models scoring >= 0.25, attaches labels
    to mature models (>= 30 days), and upserts to the Hopsworks feature store.
    """

    def __init__(self):
        """Initialize feature pipeline collaborators and relevance filtering settings."""
        self.RELEVANCE_THRESHOLD = 0.25
        self.computer = FeatureComputer()
        self.labeller = Labeller()
        self.store = HopsworksStore()
        self.ingestor = HFIngestor()

    def run(self) -> pd.DataFrame:
        """Execute the daily feature pipeline and upsert labelled feature rows."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        # Fetch new daily models
        logger.info("Fetching recently modified models from Hugging Face Hub")
        models = self.ingestor.fetch_models()
        if not models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()
        logger.info(f"Fetched {len(models)} models")

        # Re-fetch young models
        new_model_ids = {model["model_id"] for model in models}
        young_stored_ids = self.store.fetch_young_model_ids()
        refresh_ids = [model_id for model_id in young_stored_ids if model_id not in new_model_ids]

        if refresh_ids:
            logger.info(f"Re-fetching {len(refresh_ids)} young models for velocity tracking")
            models.extend(self.ingestor.fetch_models_by_id(refresh_ids))

        # Fetch prior snapshots for young models to compute download velocity
        cutoff = datetime.now(timezone.utc) - timedelta(hours=76)
        young_model_ids = [m["model_id"] for m in models if m["created_at"] and m["created_at"] >= cutoff]

        prior_snapshots = {}
        if young_model_ids:
            logger.info(f"Fetching prior snapshots for {len(young_model_ids)} young models")
            prior_snapshots = self.store.fetch_prior_snapshots(young_model_ids)

        logger.info("Computing features")
        feature_rows = self.computer.compute_features_batch(models, prior_snapshots=prior_snapshots, batch_size=64)
        df = pd.DataFrame(feature_rows)

        # Keep only models relevant to frontier topics
        before = len(df)
        df = df[df["best_topic_score"] >= self.RELEVANCE_THRESHOLD].reset_index(drop=True)
        logger.info(f"Relevance filter: {len(df)} / {before} models above {self.RELEVANCE_THRESHOLD} threshold")

        if df.empty:
            logger.warning("No models passed relevance filter, exiting")
            return df

        df = df.drop(columns=["best_topic_score"])

        logger.info("Attaching labels to mature models")
        df = self.labeller.compute_labels(df)

        logger.info(f"Pushing {len(df)} rows to Hopsworks")
        self.store.upsert(df)

        logger.info(f"Daily pipeline complete: {len(df)} models, {df['top_quartile'].notna().sum()} labelled")
        return df


if __name__ == "__main__":
    FeaturePipeline().run()
