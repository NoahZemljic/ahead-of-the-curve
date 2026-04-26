import logging
from datetime import datetime, timezone

import pandas as pd

from compute_features import FeatureComputer
from ingest import fetch_backfill_models

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
    LABEL_MATURITY_DAYS = 30

    def __init__(self):
        self._computer = FeatureComputer()

    def attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach regression and classification labels to mature models.

        Models older than LABEL_MATURITY_DAYS get:
          - download_growth_30d:  average of the HF rolling 30-day download count
                                  and an age-normalized all-time rate
                                  (downloads_all_time / age_days * 30)
          - top_quartile:         1 if download_growth_30d >= 75th percentile of cohort

        Note: The HF API does not expose historical download snapshots, so the
        live pipeline's exact day-0 vs day-30 difference is unavailable at backfill time.
        Averaging two approaches (30-day download count and age-normalized all-time rate) reduces the bias of either one alone.
        """
        now = datetime.now(timezone.utc)

        # Convert delta to days
        age_days = df["created_at"].apply(
            lambda dt: (now - dt).total_seconds() / 86400
        )

        mature_mask = age_days >= self.LABEL_MATURITY_DAYS

        df["download_growth_30d"] = None
        df["top_quartile"] = None

        # Set the download count of the past 30 days to the all time downloads
        # if all downloads happened in the past 30 days
        exact_mask = df["downloads_30d"] == df["downloads_all_time"]
        df.loc[exact_mask, "download_growth_30d"] = df.loc[exact_mask, "downloads_30d"]

        # Average 30d count and age-normalized download count for matured models
        needs_estimate = mature_mask & ~exact_mask
        if not (mature_mask.any()):
            logger.warning("No models are mature enough for labels (>= %d days old)", self.LABEL_MATURITY_DAYS)
            return df

        normalized_rate = df.loc[needs_estimate, "downloads_all_time"] / age_days[needs_estimate] * 30
        df.loc[needs_estimate, "download_growth_30d"] = round(
            (df.loc[needs_estimate, "downloads_30d"] + normalized_rate) / 2
        )

        # Compute models above the 0.75 quantile threshold
        threshold = df.loc[mature_mask, "download_growth_30d"].quantile(0.75)
        df.loc[mature_mask, "top_quartile"] = (
            df.loc[mature_mask, "download_growth_30d"] >= threshold
        ).astype(int)

        labelled_count = mature_mask.sum()
        logger.info(
            "Labelled %d / %d models (>= %d days old, 75th pctl threshold: %s)",
            labelled_count, len(df), self.LABEL_MATURITY_DAYS, threshold,
        )

        return df

    def run(self) -> pd.DataFrame:
        """Execute the full backfill pipeline: fetch > compute features > label."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info("Starting backfill for the last %d days", self.BACKFILL_DAYS)

        models = fetch_backfill_models(days=self.BACKFILL_DAYS)
        if not models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()

        logger.info("Computing features for %d models (batch_size=64)", len(models))
        feature_rows = self._computer.compute_features_batch(models, batch_size=64)

        df = pd.DataFrame(feature_rows)

        df = self.attach_labels(df)

        logger.info(
            "Backfill complete: %d models, %d with labels",
            len(df),
            df["top_quartile"].notna().sum(),
        )

        df.to_csv("backfill_output.csv", index=False)
        logger.info("Saved backfill output to backfill_output.csv")

        return df


if __name__ == "__main__":
    df = Backfill().run()
    print(f"\nBackfill result: {len(df)} rows, {df.columns.tolist()}")
    print(f"Labelled: {df['top_quartile'].notna().sum()}")
    print(f"Top quartile: {(df['top_quartile'] == 1).sum()}")
    print(f"\nSample:\n{df.head(10).to_string()}")
