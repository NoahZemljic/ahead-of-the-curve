import logging

import pandas as pd

from feature_pipeline.compute_features import FeatureComputer
from feature_pipeline.compute_labels import Labeller
from feature_pipeline.hopsworks_store import HopsworksStore
from feature_pipeline.ingest import HFIngestor

logger = logging.getLogger(__name__)

class BackfillPipeline:

    """Back-fill 90 days of frontier models from Hugging Face Hub and insert to Hopsworks.

    On first run this populates the feature store with enough labelled data
    for the training pipeline to learn from immediately, without waiting 30 days
    for live labels to accumulate.

    Fetches models for targeted topics (robotics, multimodal reasoning, SLMs),
    computes features, filters by semantic relevance (>= 0.25), attaches
    labels to models that have matured (>= 30 days old), and inserts to Hopsworks.
    """

    def __init__(self):
        """Initialize backfill settings and pipeline collaborators."""
        self.BACKFILL_DAYS = 90
        self.RELEVANCE_THRESHOLD = 0.25
        self.computer = FeatureComputer()
        self.labeller = Labeller()
        self.ingestor = HFIngestor()
        self.store = HopsworksStore()

    def run(self) -> pd.DataFrame:
        """Execute the full backfill pipeline: fetch > compute features > filter > label > insert."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info(f"Starting backfill for the last {self.BACKFILL_DAYS} days")

        # Fetch models from the last 90 days for backfill
        backfill_models = self.ingestor.fetch_models_backfill(since_days=self.BACKFILL_DAYS)
        if not backfill_models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()
        logger.info(f"Fetched {len(backfill_models)} models")

        # Compute features in batches
        logger.info(f"Computing features for {len(backfill_models)} models (batch_size=64)")
        features = self.computer.compute_features_batch(backfill_models, batch_size=64)
        features_df = pd.DataFrame(features)

        # Keep only models relevant to frontier topics
        before = len(features_df)
        features_df = features_df[features_df["best_topic_score"] >= self.RELEVANCE_THRESHOLD].reset_index(drop=True)
        logger.info(f"Relevance filter: {len(features_df)} / {before} models above {self.RELEVANCE_THRESHOLD} threshold")

        if features_df.empty:
            logger.warning("No models passed relevance filter, exiting")
            return features_df

        # Best score was only needed for relevance filtering
        features_df = features_df.drop(columns=["best_topic_score"])

        # Compute labels for the backfill models that are 30+ days old
        features_df = self.labeller.compute_labels(features_df)

        logger.info(f"Pushing {len(features_df)} rows to Hopsworks")
        self.store.insert(features_df)

        labelled = features_df['top_quartile'].notna().sum()
        logger.info(f"Backfill complete: {len(features_df)} models, {labelled} labelled")
        return features_df

if __name__ == "__main__":
    BackfillPipeline().run()
