import logging

import pandas as pd

from compute_features import FeatureComputer
from compute_labels import Labeller
from ingest import HFIngestor

logger = logging.getLogger(__name__)


class Backfill:
    """Back-fill 90 days of frontier models from Hugging Face Hub.

    On first run this populates the feature store with enough labelled data
    for the training pipeline to learn from immediately, without waiting 30 days
    for live labels to accumulate.

    Fetches models for targeted topics (robotics, multimodal reasoning, SLMs),
    computes features, filters by semantic relevance (>= 0.25), and attaches
    labels to models that have matured (>= 30 days old).
    """

    def __init__(self):
        self.BACKFILL_DAYS = 90
        self.RELEVANCE_THRESHOLD = 0.25
        self._computer = FeatureComputer()
        self._labeller = Labeller()
        self._ingestor = HFIngestor()

    def run(self) -> pd.DataFrame:
        """Execute the full backfill pipeline: fetch > compute features > filter > label."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info(f"Starting backfill for the last {self.BACKFILL_DAYS} days")

        # Fetch models from the last 90 days
        backfill_models = self._ingestor.fetch_models_backfill(since_days=self.BACKFILL_DAYS)
        if not backfill_models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()

        # Compute features in batches of 64
        logger.info(f"Computing features for {len(backfill_models)} models (batch_size=64)")
        feature_rows = self._computer.compute_features_batch(backfill_models, batch_size=64)
        df = pd.DataFrame(feature_rows)

        # Discard models with a relevance threshold below 0.25
        before = len(df)
        df = df[df["best_topic_score"] >= self.RELEVANCE_THRESHOLD].reset_index(drop=True)
        logger.info(f"Relevance filter: {len(df)} / {before} models above {self.RELEVANCE_THRESHOLD} threshold")

        if df.empty:
            logger.warning("No models passed relevance filter, exiting")
            return df

        df = df.drop(columns=["best_topic_score"])

        df = self._labeller.compute_labels(df)

        logger.info(f"Backfill complete: {len(df)} models, {df['top_quartile'].notna().sum()} with labels")

        df.to_csv("backfill_output.csv", index=False)
        logger.info("Saved backfill output to backfill_output.csv")

        return df


if __name__ == "__main__":
    df = Backfill().run()
    print(f"\nBackfill result: {len(df)} rows, {df.columns.tolist()}")
    print(f"Labelled: {df['top_quartile'].notna().sum()}")
    print(f"Top quartile: {(df['top_quartile'] == 1).sum()}")
    print(f"\nSample:\n{df.head(10).to_string()}")
