# Ahead of the Curve

**Noah Zemljic**

Thousands of AI models reach Hugging Face every day; the important ones only surface once they are already trending. Ahead of the Curve is an end-to-end MLOps system that spots high-potential releases within hours of upload, predicting each model's 30-day download count and whether it will reach the top quartile of its cohort. It focuses on three frontier niches: robotics, small language models and multimodal reasoning. A daily feature pipeline, weekly XGBoost training, and bi-hourly batch inference run automatically across Hopsworks, DagsHub MLflow and Google Cloud Run, surfacing live predictions in a dashboard.

![Ahead of the Curve — high-level architecture](../architecture.png)