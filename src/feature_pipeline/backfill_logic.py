import logging

import pandas as pd

from compute_features import FeatureComputer
from ingest import fetch_models
from feature_pipeline.compute_labels import Labeller

logger = logging.getLogger(__name__)


class Backfill:
    """Back-fill 90 days of historical robotics models from Hugging Face Hub.

    On first run this populates the feature store with enough labelled data
    for the training pipeline to learn from immediately, without waiting 30 days
    for live labels to accumulate.

    For each model created in the last 90 days:
      - Features are computed (semantic relevance, momentum, metadata).
      - If the model is >= 30 days old, labels are attached:
          * download_growth_30d  — downloads_all_time as a proxy for total adoption
          * top_quartile         — whether download growth lands in the top 25% of the cohort
    """

    BACKFILL_DAYS = 90

    def __init__(self):
        self._computer = FeatureComputer()
        self._labeller = Labeller()

    def run(self) -> pd.DataFrame:
        """Execute the full backfill pipeline: fetch > compute features > label."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info(f"Starting backfill for the last {self.BACKFILL_DAYS} days")

        models = fetch_models(since_days=self.BACKFILL_DAYS)
        if not models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()

        logger.info(f"Computing features for {len(models)} models (batch_size=64)")
        feature_rows = self._computer.compute_features_batch(models, batch_size=64)

        df = pd.DataFrame(feature_rows)

        df = self._labeller.attach_labels(df)

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
