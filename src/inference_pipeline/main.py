import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel

from preprocessing import PreProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ModelFeatures(BaseModel):
    relevance_robotics: float
    relevance_slm: float
    relevance_multimodal: float
    age_hours: float
    download_velocity_24h: float = 0.0
    download_velocity_72h: float = 0.0
    likes: int
    trending_score: float
    downloads_30d: int
    downloads_all_time: int
    has_paper_tag: int
    tag_count: int
    best_topic: str


class Prediction(BaseModel):
    trend_score: float
    top_quartile_prob: float
    top_quartile: int


class ModelPrediction(BaseModel):
    model_id: str
    predicted_at: datetime
    downloads_30d_pred: float
    top_quartile_prob: float
    top_quartile_pred: int


def load_model_from_gcs(bucket_name: str, model_type: str):
    """Download and deserialise a joblib model from GCS into memory."""
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(f"models/{model_type}/model.joblib")
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    return joblib.load(buffer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    logger.info("Loading models from GCS")
    app.state.regressor = load_model_from_gcs(bucket_name, "regressor")
    app.state.classifier = load_model_from_gcs(bucket_name, "classifier")
    app.state.preprocessor = PreProcessor()
    logger.info("Models ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions", response_model=list[ModelPrediction])
def get_predictions(limit: int = 50):
    try:
        bucket_name = os.environ["GCS_BUCKET_NAME"]
        client = storage.Client()
        blob = client.bucket(bucket_name).blob("predictions/latest.json")
        records = json.loads(blob.download_as_text())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return records[:limit]


@app.post("/predict", response_model=Prediction)
def predict(features: ModelFeatures):
    try:
        df = app.state.preprocessor.process(pd.DataFrame([features.model_dump()]))
        trend_score = float(app.state.regressor.predict(df)[0])
        top_quartile_prob = float(app.state.classifier.predict_proba(df)[0][1])
        top_quartile = int(app.state.classifier.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Prediction(
        trend_score=trend_score,
        top_quartile_prob=top_quartile_prob,
        top_quartile=top_quartile,
    )
