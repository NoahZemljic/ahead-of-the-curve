import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class PreProcessor:
    """Prepare raw feature-store rows for model training.

    One-hot encodes `best_topic`, imputes missing velocity features,
    scales numeric columns, and splits into train/validation/test sets.
    """

    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
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
        self._topic_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._scaler = StandardScaler()
        self._test_size = test_size
        self._val_size = val_size
        self._random_state = random_state

    def _encode_topic(self, topic_series: pd.Series) -> pd.DataFrame:
        """Fit and transform the `best_topic` column into one-hot columns."""
        values = topic_series.fillna("unknown").values.reshape(-1, 1)
        encoded = self._topic_encoder.fit_transform(values)

        topic_feature_names = [
            f"topic_{name}" for name in self._topic_encoder.categories_[0]
        ]

        return pd.DataFrame(encoded, columns=topic_feature_names, index=topic_series.index)

    def process(self, df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
        """Clean, encode, scale, and split a raw feature-store DataFrame.

        Returns a dict with keys: X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_clf_train, y_clf_val, y_clf_test.
        """
        df = df.copy()

        # Extract targets
        y_reg = df["download_growth_30d"]
        y_clf = df["top_quartile"]

        # Drop identifiers, timestamps and targets
        drop_cols = [
            "model_id",
            "created_at",
            "snapshot_date",
            "download_growth_30d",
            "top_quartile",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # One-hot encode best_topic
        topic_encodings = self._encode_topic(df["best_topic"])
        df = pd.concat([df.drop(columns=["best_topic"]), topic_encodings], axis=1)

        # Impute missing velocity features with 0
        df["download_velocity_24h"] = df["download_velocity_24h"].fillna(0)
        df["download_velocity_72h"] = df["download_velocity_72h"].fillna(0)

        # Split before scaling to avoid data leakage
        X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
            df, y_reg, y_clf,
            test_size=self._test_size + self._val_size,
            random_state=self._random_state,
        )

        relative_val_size = self._val_size / (self._test_size + self._val_size)

        X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
            X_temp, y_reg_temp, y_clf_temp,
            test_size=1 - relative_val_size,
            random_state=self._random_state,
        )

        # Scale numeric features
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in X_train.columns]
        X_train[numeric_cols] = self._scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = self._scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = self._scaler.transform(X_test[numeric_cols])

        logger.info(
            f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_reg_train": y_reg_train, "y_reg_val": y_reg_val, "y_reg_test": y_reg_test,
            "y_clf_train": y_clf_train, "y_clf_val": y_clf_val, "y_clf_test": y_clf_test,
        }
