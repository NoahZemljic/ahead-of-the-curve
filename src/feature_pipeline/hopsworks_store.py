import logging
import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class HopsworksStore:
    """Wrapper around Hopsworks feature store for upserting model feature data.

    Handles connection management, DataFrame type coercion, and feature group
    creation. Shared by both the daily pipeline and the backfill manager.
    """

    FEATURE_GROUP_NAME = "frontier_models_features"
    FEATURE_GROUP_VERSION = 1

    def __init__(self):
        self._fs = None

    def _get_feature_store(self):
        """Connect to Hopsworks and return the feature store handle (cached)."""
        if self._fs is None:
            api_key = os.getenv("HOPSWORKS_API_KEY")
            project = hopsworks.login(api_key_value=api_key)
            self._fs = project.get_feature_store()
        return self._fs

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce column types for Hopsworks compatibility."""
        df = df.copy()

        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)

        df["has_paper_tag"] = df["has_paper_tag"].astype(int)

        for col in [
            "top_quartile",
            "download_growth_30d",
            "download_velocity_24h",
            "download_velocity_72h",
            "age_hours",
            "relevance_robotics",
            "relevance_slm",
            "relevance_multimodal",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["download_velocity_24h"] = df["download_velocity_24h"].fillna(0)
        df["download_velocity_72h"] = df["download_velocity_72h"].fillna(0)

        df["best_topic"] = df["best_topic"].astype(str)

        return df

    def upsert(self, df: pd.DataFrame) -> None:
        """Upsert a DataFrame into the Hopsworks feature group.

        Creates the feature group on first run with primary key (model_id, snapshot_date)
        and an event_time column for time-travel queries.
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to upsert")
            return

        df = self._prepare_dataframe(df)
        fs = self._get_feature_store()

        fg = fs.get_or_create_feature_group(
            name=self.FEATURE_GROUP_NAME,
            version=self.FEATURE_GROUP_VERSION,
            description="Frontier model features with labels for trend prediction",
            primary_key=["model_id", "snapshot_date"],
            event_time="snapshot_date",
        )

        fg.insert(df)
        logger.info(f"Upserted {len(df)} rows to feature group '{self.FEATURE_GROUP_NAME}'")
