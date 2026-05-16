import logging

import pandas as pd
from dotenv import load_dotenv

from common.hopsworks_client import HopsworksFeatureStoreClient

load_dotenv()

logger = logging.getLogger(__name__)


class PredictionsStore(HopsworksFeatureStoreClient):
    """Hopsworks feature group for persisting inference predictions.

    Predictions are written by the hourly batch inference worker and read
    by the serving API. The feature group acts as an audit trail and enables
    connecting predictions to ground-truth labels once models mature.
    """

    def __init__(self):
        super().__init__()
        self.FEATURE_GROUP_NAME = "frontier_models_predictions"
        self.FEATURE_GROUP_VERSION = 1

    def insert(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("Empty DataFrame, nothing to insert")
            return

        df = df.copy()
        df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
        df["downloads_30d_pred"] = df["downloads_30d_pred"].astype(float)
        df["top_quartile_prob"] = df["top_quartile_prob"].astype(float)
        df["top_quartile_pred"] = df["top_quartile_pred"].astype(int)

        fs = self.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=self.FEATURE_GROUP_NAME,
            version=self.FEATURE_GROUP_VERSION,
            description="Hourly frontier model predictions for trend scoring",
            primary_key=["model_id", "predicted_at"],
            event_time="predicted_at",
        )
        fg.insert(df)
        logger.info(f"Inserted {len(df)} prediction rows to '{self.FEATURE_GROUP_NAME}'")

    def fetch_latest(self, limit: int = 50) -> pd.DataFrame:
        """Return the most recent `limit` predictions ordered by predicted_at descending."""
        fs = self.get_feature_store()

        try:
            fg = fs.get_feature_group(
                name=self.FEATURE_GROUP_NAME,
                version=self.FEATURE_GROUP_VERSION,
            )
        except Exception:
            return pd.DataFrame()

        df = fg.select_all().read()
        if df.empty:
            return df

        df["predicted_at"] = pd.to_datetime(df["predicted_at"], utc=True)
        return (
            df.sort_values("predicted_at", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )
