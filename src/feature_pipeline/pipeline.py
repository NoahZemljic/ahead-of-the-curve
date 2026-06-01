import logging

import pandas as pd

from feature_pipeline.compute_features import FeatureComputer
from feature_pipeline.compute_labels import Labeller
from feature_pipeline.hopsworks_store import HopsworksStore
from feature_pipeline.ingest import HFIngestor

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Daily feature pipeline: fetch, score, filter, label, push to Hopsworks.

    Runs once per day at 00:00 UTC. Fetches all recently modified models from
    the Hub (regardless of tag), computes features including semantic relevance
    against frontier topics, filters to models scoring >= 0.25, fetches mature models, updates mature model download counts, attaches labels
    to mature models (>= 30 days), and inserts to the Hopsworks feature store.
    """

    def __init__(self):
        """Initialize feature pipeline collaborators and relevance filtering settings."""
        self.RELEVANCE_THRESHOLD = 0.25
        self.computer = FeatureComputer()
        self.labeller = Labeller()
        self.store = HopsworksStore()
        self.ingestor = HFIngestor()

    def run(self) -> pd.DataFrame:
        """Execute the daily feature pipeline and insert labelled feature rows."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        # Fetch new daily models
        logger.info("Fetching recently modified models from Hugging Face Hub")
        models = self.ingestor.fetch_models()
        if not models:
            logger.warning("No models found, exiting")
            return pd.DataFrame()
        logger.info(f"Fetched {len(models)} models")

        new_model_ids = {model["model_id"] for model in models}

        # Get models eligible for download velocity computation
        young_stored_ids = self.store.fetch_young_model_ids()
        ids_to_be_refreshed = [model_id for model_id in young_stored_ids if model_id not in new_model_ids]

        if ids_to_be_refreshed:
            logger.info(f"Re-fetching {len(ids_to_be_refreshed)} young models for velocity tracking")
            models.extend(self.ingestor.fetch_models_by_id(ids_to_be_refreshed))

        young_model_ids = list(young_stored_ids)

        # Fetch prior snapshots for eligible models
        prior_snapshots = {}
        if young_model_ids:
            logger.info(f"Fetching prior snapshots for {len(young_model_ids)} young models")
            prior_snapshots = self.store.fetch_prior_model_snapshots(young_model_ids)

        # Compute features in batches
        logger.info("Computing features")
        features = self.computer.compute_features_batch(models, prior_snapshots=prior_snapshots, batch_size=64)
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

        # Fetch mature unlabelled models
        mature_df = self.store.fetch_mature_models()
        if not mature_df.empty:
            logger.info(f"Labelling {len(mature_df)} mature unlabelled models")

            # Get current download counts for mature models
            current_models = self.ingestor.fetch_models_by_id(mature_df["model_id"].tolist())

            temp = {model["model_id"]: model for model in current_models}

            # Update mature model row with downloads in last 30 days
            mature_df["downloads_30d"] = mature_df["model_id"].map(
                {mid: m["downloads_30d"] for mid, m in temp.items()}
            )

            # Update mature model row with all time downloads
            mature_df["downloads_all_time"] = mature_df["model_id"].map(
                {mid: m["downloads_all_time"] for mid, m in temp.items()}
            )

            # Compute labels for unlabelled mature model
            mature_df = self.labeller.compute_labels(mature_df)

        # Combine new / young models with mature models and insert to hopsworks
        all_processed_models = pd.concat([features_df, mature_df], ignore_index=True)

        logger.info(f"Pushing {len(all_processed_models)} rows to Hopsworks")
        self.store.insert(all_processed_models)

        labelled = all_processed_models["top_quartile"].notna().sum()
        logger.info(f"Daily pipeline complete: {len(all_processed_models)} models, {labelled} labelled")
        return all_processed_models

if __name__ == "__main__":
    FeaturePipeline().run()
