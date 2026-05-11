import logging

import pandas as pd

from backfill_logic import Backfill
from feature_pipeline.hopsworks_store import HopsworksStore

logger = logging.getLogger(__name__)


class BackfillManager:
    """Orchestrates the backfill pipeline and upserts results to Hopsworks."""

    def __init__(self):
        self._backfill = Backfill()
        self._store = HopsworksStore()

    def run(self) -> pd.DataFrame:
        """Execute the full backfill pipeline and push results to Hopsworks."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        logger.info("Running backfill pipeline")
        df = self._backfill.run()

        if df.empty:
            logger.warning("Backfill returned no data, skipping Hopsworks upsert")
            return df

        logger.info(f"Pushing {len(df)} rows to Hopsworks")
        self._store.upsert(df)
        logger.info("Backfill complete")
        return df


if __name__ == "__main__":
    BackfillManager().run()
