import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from common.hopsworks_client import HopsworksFeatureStoreClient

load_dotenv()

logger = logging.getLogger(__name__)


class HopsworksStore(HopsworksFeatureStoreClient):
    """Wrapper around Hopsworks feature store for inserting model feature data.

    Handles connection management, DataFrame type coercion, and feature group
    creation. Shared by both the daily pipeline and the backfill manager.
    """

    def __init__(self):
        """Initialize the shared Hopsworks feature store connection state."""
        super().__init__()

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce column types for Hopsworks compatibility."""
        df = df.copy()

        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)

        df["has_paper_tag"] = df["has_paper_tag"].astype(int)

        for col in ["downloads_30d", "downloads_all_time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

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

    def fetch_young_model_ids(self, max_age_hours: int = 76) -> list[str]:
        """Return model_ids created within the last max_age_hours from the feature store.

        Used to identify models that still need velocity snapshots even if
        they no longer appear in the daily fetch_models() results.
        """
        fs = self.get_feature_store()

        try:
            fg = fs.get_feature_group(
                name=self.FEATURE_GROUP_NAME,
                version=self.FEATURE_GROUP_VERSION,
            )
        except Exception:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        df = fg.select(["model_id", "created_at"]) \
             .filter(fg.created_at >= cutoff) \
             .read()

        if df.empty:
            return []

        return df["model_id"].unique().tolist()

    def fetch_mature_models(self, min_age_days: int = 30) -> pd.DataFrame:
        """Fetch the latest stored feature row for each mature unlabelled model.

        Returns all feature columns so only the download counts need to be
        refreshed before label computation.
        """
        fs = self.get_feature_store()

        try:
            fg = fs.get_feature_group(
                name=self.FEATURE_GROUP_NAME,
                version=self.FEATURE_GROUP_VERSION,
            )
        except Exception:
            return pd.DataFrame()

        cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)
        df = fg.select_all().filter(fg.created_at <= cutoff).read()

        if df.empty:
            return df

        df = df[df["top_quartile"].isna()]

        if df.empty:
            return df

        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)
        return df.sort_values("snapshot_date").groupby("model_id").last().reset_index()

    def fetch_prior_model_snapshots(self, model_ids: list[str]) -> dict[str, list[dict]]:
        """Fetch prior snapshots from the feature store for velocity computation."""
        if not model_ids:
            return {}

        fs = self.get_feature_store()

        try:
            fg = fs.get_feature_group(
                name=self.FEATURE_GROUP_NAME,
                version=self.FEATURE_GROUP_VERSION,
            )
        except Exception:
            logger.info("Feature group not found, no prior snapshots available")
            return {}

        # Fetch previous model versions
        df = fg.select(["model_id", "snapshot_date", "downloads_30d"]) \
             .filter(fg.model_id.isin(model_ids)) \
             .read()

        if df.empty:
            return {}

        snapshots = {}

        for model_id, group in df.groupby("model_id"):
            snapshots[model_id] = [
                {
                    "snapshot_date": row["snapshot_date"].isoformat(),
                    "downloads_30d": row["downloads_30d"],
                }
                for _, row in group.iterrows()
            ]

        logger.info(f"Fetched prior snapshots for {len(snapshots)} / {len(model_ids)} models")
        return snapshots

    def insert(self, df: pd.DataFrame) -> None:
        """Insert a DataFrame into the Hopsworks feature group.

        Creates the feature group on first run with primary key (model_id, snapshot_date)
        and an event_time column for time-travel queries.
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to insert")
            return

        df = self.prepare_dataframe(df)
        fs = self.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name=self.FEATURE_GROUP_NAME,
            version=self.FEATURE_GROUP_VERSION,
            description="Frontier model features with labels for trend prediction",
            primary_key=["model_id", "snapshot_date"],
            event_time="snapshot_date",
        )

        fg.insert(df)
        logger.info(f"Inserted {len(df)} rows to feature group '{self.FEATURE_GROUP_NAME}'")
