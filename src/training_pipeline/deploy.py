import logging
import os
import tempfile
import time

import joblib
from google.cloud import run_v2, storage

logger = logging.getLogger(__name__)

class GCPDeployer:
    """Upload champion models to GCS and trigger a new Cloud Run service revision."""

    def __init__(self):
        """Resolve GCP config from environment and initialise GCS and Cloud Run clients."""
        self.bucket_name = os.environ["GCS_BUCKET_NAME"]
        self.project_id = os.environ["GCP_PROJECT_ID"]
        self.region = os.environ.get("GCP_REGION", "europe-west6")
        self.service_name = os.environ.get("CLOUD_RUN_SERVICE", "inference-pipeline")
        self.storage_client = storage.Client(project=self.project_id)
        self.run_client = run_v2.ServicesClient()

    def deploy(self, pipeline, model_type):
        """Upload a promoted pipeline to GCS and trigger a new Cloud Run revision."""
        gcs_uri = self.upload_model(pipeline, model_type)
        logger.info(f"Uploaded {model_type} to {gcs_uri}")
        self.update_service(model_type)

    def upload_model(self, pipeline, model_type):
        """Serialise the pipeline to a temp file and upload it to the fixed GCS path for model_type."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(f"models/{model_type}/model.joblib")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.joblib")
            joblib.dump(pipeline, path)
            blob.upload_from_filename(path)
        return f"gs://{self.bucket_name}/models/{model_type}/model.joblib"

    def update_service(self, model_type):
        """Forces a new Cloud Run revision by mutating the update env variable so the inference service reloads the model from GCS.
        """
        name = (
            f"projects/{self.project_id}/locations/{self.region}"
            f"/services/{self.service_name}"
        )
        try:
            service = self.run_client.get_service(name=name)
        except Exception as exc:
            logger.warning(f"Cloud Run service not found, skipping update: {exc}")
            return

        container = service.template.containers[0]
        env_vars = {e.name: e.value for e in container.env}
        # Mutating this env var signals Cloud Run that a new revision is needed
        env_vars[f"{model_type.upper()}_UPDATED_AT"] = str(int(time.time()))
        container.env = [run_v2.EnvVar(name=k, value=v) for k, v in env_vars.items()]

        try:
            operation = self.run_client.update_service(service=service)
            operation.result()
            logger.info(f"Cloud Run service updated for {model_type}")
        except Exception as exc:
            logger.warning(f"Failed to update Cloud Run service: {exc}")
