import io
import logging
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage

from feature_pipeline.compute_features import FeatureComputer
from feature_pipeline.ingest import HFIngestor
from hopsworks_store import PredictionsStore
from preprocessing import PreProcessor

load_dotenv()

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Hourly batch inference: fetch new HF models, score them, write predictions to Hopsworks."""

    def __init__(self):
        self.FETCH_WINDOW_HOURS = 2
        self.RELEVANCE_THRESHOLD = 0.25
        self.ingestor = HFIngestor()
        self.computer = FeatureComputer()
        self.preprocessor = PreProcessor()
        self.store = PredictionsStore()
        bucket = os.environ["GCS_BUCKET_NAME"]
        self.regressor = self.load_model_from_gcs(bucket, "regressor")
        self.classifier = self.load_model_from_gcs(bucket, "classifier")

    def load_model_from_gcs(self, bucket_name: str, model_type: str):
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(f"models/{model_type}/model.joblib")
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return joblib.load(buffer)

    def run(self) -> pd.DataFrame:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

        # Fetch recently created models
        logger.info(f"Fetching models created in the last {self.FETCH_WINDOW_HOURS} hours")
        recent_models = self.ingestor.fetch_recent_models(hours=self.FETCH_WINDOW_HOURS)

        if not recent_models:
            logger.warning("No new models found, exiting")
            return pd.DataFrame()

        logger.info(f"Fetched {len(recent_models)} models")

        # Compute features for models
        features = self.computer.compute_features_batch(recent_models, batch_size=64)
        features_df = pd.DataFrame(features)

        # Keep only models relevant to frontier topics
        before = len(features_df)
        features_df = features_df[
            features_df["best_topic_score"] >= self.RELEVANCE_THRESHOLD
        ].reset_index(drop=True)

        logger.info(
            f"Relevance filter: {len(features_df)} / {before} models above "
            f"{self.RELEVANCE_THRESHOLD} threshold"
        )

        if features_df.empty:
            logger.warning("No models passed relevance filter, exiting")
            return features_df


        model_ids = features_df["model_id"].tolist()
        predicted_at = features_df["snapshot_date"].iloc[0]

        inference_df = self.preprocessor.process(features_df)

        downloads_30d_pred = self.regressor.predict(inference_df).tolist()
        top_quartile_prob = self.classifier.predict_proba(inference_df)[:, 1].tolist()
        top_quartile_pred = self.classifier.predict(inference_df).tolist()

        predictions_df = pd.DataFrame({
            "model_id": model_ids,
            "best_topic": features_df["best_topic"].tolist(),
            "predicted_at": predicted_at,
            "downloads_30d_pred": downloads_30d_pred,
            "top_quartile_prob": top_quartile_prob,
            "top_quartile_pred": top_quartile_pred,
        })

        logger.info(f"Writing {len(predictions_df)} predictions to Hopsworks")
        self.store.insert(predictions_df)
        self._write_latest_to_gcs(predictions_df, os.environ["GCS_BUCKET_NAME"])

        top = (predictions_df["top_quartile_pred"] == 1).sum()
        logger.info(f"Done: {len(predictions_df)} predictions, {top} predicted top-quartile")
        return predictions_df

    def _write_latest_to_gcs(self, predictions_df: pd.DataFrame, bucket_name: str) -> None:
        payload = predictions_df.to_json(orient="records", date_format="iso")
        client = storage.Client()
        client.bucket(bucket_name).blob("predictions/latest.json").upload_from_string(
            payload, content_type="application/json"
        )
        logger.info(f"Wrote latest predictions to gs://{bucket_name}/predictions/latest.json")


if __name__ == "__main__":
    InferencePipeline().run()
