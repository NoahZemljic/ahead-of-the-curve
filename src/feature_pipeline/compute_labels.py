import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class Labeller:
    """Attach regression and classification labels to mature models (>= 30 days old).

    Shared by both the backfill pipeline and the daily orchestration.

    Models older than LABEL_MATURITY_DAYS get:
      - download_growth_30d: For young models whose entire download history
            falls within 30 days, the rolling count already represent the correct value.
            For older models the rolling count only covers the most recent window, so it
            is averaged with the all-time downloads scaled down to a 30-day
            rate to better approximate early adoption.
      - top_quartile: whether the model's download growth lands in the top
            25 percent of its cohort.
    """

    LABEL_MATURITY_DAYS = 30

    def attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        now = datetime.now(timezone.utc)

        df["download_growth_30d"] = None
        df["top_quartile"] = None

        age_days = df["created_at"].apply(
            lambda dt: (now - dt).total_seconds() / 86400
        )

        # Only label models that have been around for at least 30 days
        mature = age_days >= self.LABEL_MATURITY_DAYS
        if not mature.any():
            logger.warning(f"No models are mature enough for labels (>= {self.LABEL_MATURITY_DAYS} days old)")
            return df

        # Estimate 30-day download growth for each mature model
        for i in df.loc[mature].index:
            downloads_30d = df.at[i, "downloads_30d"]
            downloads_all = df.at[i, "downloads_all_time"]

            if downloads_30d == downloads_all:
                # All downloads happened within the last 30 days, so the
                # rolling count is already the exact 30-day growth
                df.at[i, "download_growth_30d"] = downloads_30d
            else:
                # Average the last 30 days of downloads with the all-time
                # downloads scaled to a 30-day rate for a better estimate.
                daily_rate = downloads_all / age_days[i]
                normalized_30d = daily_rate * 30
                df.at[i, "download_growth_30d"] = round((downloads_30d + normalized_30d) / 2)

        # Mark models in the top 25% of download growth as top quartile
        threshold = df.loc[mature, "download_growth_30d"].quantile(0.75)
        df.loc[mature, "top_quartile"] = (
            df.loc[mature, "download_growth_30d"] >= threshold
        ).astype(int)

        labelled_count = mature.sum()
        logger.info(
            f"Labelled {labelled_count} / {len(df)} models (>= {self.LABEL_MATURITY_DAYS} days old, 75th pctl threshold: {threshold})",
        )

        return df
