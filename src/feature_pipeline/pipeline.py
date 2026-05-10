import logging

import pandas as pd

from compute_features import FeatureComputer
from compute_labels import Labeller
from hopsworks_store import HopsworksStore
from ingest import fetch_models

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Daily feature pipeline: fetch, score, filter, label, push to Hopsworks.

    Runs once per day at 00:00 UTC. Fetches all recently modified models from
    the Hub (regardless of tag), computes features including semantic relevance
    against frontier topics, filters to models scoring >= 0.25, attaches labels
    to mature models (>= 30 days), and upserts to the Hopsworks feature store.
    """

    RELEVANCE_THRESHOLD = 0.25

    def __init__(self):
        self._computer = FeatureComputer()
        self._labeller = Labeller()
        self._store = HopsworksStore()

    def run(self) -> pd.DataFrame:
        """Execute the daily feature pipeline."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        # Fetch new daily models
        logger.info("Fetching recently modified models from Hugging Face Hub")
        models = fetch_models()
        if not models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()
        logger.info(f"Fetched {len(models)} models")

        logger.info("Computing features")
        feature_rows = self._computer.compute_features_batch(models, batch_size=64)
        df = pd.DataFrame(feature_rows)

        # Keep only models relevant to frontier topics
        before = len(df)
        df = df[df["best_topic_score"] >= self.RELEVANCE_THRESHOLD].reset_index(drop=True)
        logger.info(f"Relevance filter: {len(df)} / {before} models above {self.RELEVANCE_THRESHOLD} threshold")

        if df.empty:
            logger.warning("No models passed relevance filter, exiting")
            return df

        logger.info("Attaching labels to mature models")
        df = self._labeller.attach_labels(df)

        logger.info(f"Pushing {len(df)} rows to Hopsworks")
        self._store.upsert(df)

        logger.info(f"Daily pipeline complete: {len(df)} models, {df['top_quartile'].notna().sum()} labelled")
        return df


if __name__ == "__main__":
    FeaturePipeline().run()
