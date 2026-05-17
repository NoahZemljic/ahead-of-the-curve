import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

TOPICS = ["multimodal_reasoning", "robotics", "slm", "unknown"]


class PreProcessor:
    """Prepare raw feature rows for inference.

    Mirrors the training PreProcessor exactly: drops metadata columns,
    one-hot encodes `best_topic`, and imputes missing velocity features.
    The encoder is pre-fit on the fixed topic categories so transform is
    safe regardless of which topics appear in a given batch.
    """

    def __init__(self):
        self.topic_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def encode_topic(self, topic_series: pd.Series) -> pd.DataFrame:
        """Fit and transform the `best_topic` column into one-hot columns."""

        values = topic_series.fillna("unknown").values.reshape(-1, 1)
        encoded = self.topic_encoder.fit_transform(values)

        topic_feature_names = [
            f"topic_{name}" for name in self.topic_encoder.categories_[0]
        ]
        return pd.DataFrame(encoded, columns=topic_feature_names, index=topic_series.index)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop metadata, one-hot encode best_topic, and impute velocity nulls."""
        df = df.copy()

        drop_cols = ["model_id", "created_at", "snapshot_date",
                     "best_topic_score", "download_growth_30d", "top_quartile",
                     "downloads_30d", "downloads_all_time"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        topic_encodings = self.encode_topic(df["best_topic"])
        df = pd.concat([df.drop(columns=["best_topic"]), topic_encodings], axis=1)

        df["download_velocity_24h"] = df["download_velocity_24h"].fillna(0)
        df["download_velocity_72h"] = df["download_velocity_72h"].fillna(0)

        return df
