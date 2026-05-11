import logging

import pandas as pd

from common.hopsworks_client import HopsworksFeatureStoreClient

logger = logging.getLogger(__name__)

class TrainingDataLoader(HopsworksFeatureStoreClient):
    """Read labelled training data from the Hopsworks feature store.

    Inherits the shared connection manager so it can read the same feature
    group without depending on the feature pipeline's write logic.
    """

    def __init__(self):
        super().__init__()

    def load(self) -> pd.DataFrame:
        fs = self.get_feature_store()
        fg = fs.get_feature_group(
            name=self.FEATURE_GROUP_NAME,
            version=self.FEATURE_GROUP_VERSION,
        )

        df = fg.select_all().read()

        # Discard immature models
        df = df.dropna(subset=["download_growth_30d", "top_quartile"])

        if df.empty:
            return df

        logger.info(f"Found {len(df)} labelled rows for training")

        # Coerce types
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], utc=True)
        df["has_paper_tag"] = df["has_paper_tag"].astype(int)
        df["top_quartile"] = df["top_quartile"].astype(int)

        numeric_cols = [
            "relevance_robotics",
            "relevance_slm",
            "relevance_multimodal",
            "age_hours",
            "download_velocity_24h",
            "download_velocity_72h",
            "likes",
            "trending_score",
            "downloads_30d",
            "downloads_all_time",
            "tag_count",
            "download_growth_30d",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

if __name__ == "__main__":
    data_loader = TrainingDataLoader()

    data = data_loader.load()

    print(f"{len(data)} rows")
