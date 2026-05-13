import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class PreProcessor:
    """Prepare raw feature-store rows for model training.

    One-hot encodes `best_topic` and imputes missing velocity features.
    """

    def __init__(self):
        """Initialize feature lists and encoders used during preprocessing."""
        self.NUMERIC_FEATURES = [
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
        ]
        self.topic_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def encode_topic(self, topic_series: pd.Series) -> pd.DataFrame:
        """Fit and transform the `best_topic` column into one-hot columns."""
        values = topic_series.fillna("unknown").values.reshape(-1, 1)
        encoded = self.topic_encoder.fit_transform(values)

        topic_feature_names = [
            f"topic_{name}" for name in self.topic_encoder.categories_[0]
        ]

        return pd.DataFrame(encoded, columns=topic_feature_names, index=topic_series.index)

    def process(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Clean, encode, and impute a raw feature-store DataFrame for training."""
        df = df.copy()

        drop_cols = [
            "model_id",
            "created_at",
            "snapshot_date",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        topic_encodings = self.encode_topic(df["best_topic"])
        df = pd.concat([df.drop(columns=["best_topic"]), topic_encodings], axis=1)

        df["download_velocity_24h"] = df["download_velocity_24h"].fillna(0)
        df["download_velocity_72h"] = df["download_velocity_72h"].fillna(0)

        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]

        return {
            "df": df,
            "numeric_cols": numeric_cols,
        }
