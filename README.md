# Ahead of the Curve

Identifying Frontier Models Before the Crowd Does

## Project Structure

```
ahead-of-the-curve/
├── .github/                   # GitHub Actions workflows
├── dashboard/                 # Gradio app
├── docker/                    # Dockerfiles
├── documents/                 # Project documentation
├── notebooks/                 # EDA / prototyping
├── src/
│   ├── common/                # Shared pipeline utilities
│   ├── feature_pipeline/      # Data ingestion + feature engineering
│   ├── inference_pipeline/    # Prediction serving
│   └── training_pipeline/     # Model training + evaluation + deployment
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── .python-version            # Python version pin (3.11)
├── README.md                  # You are reading this
├── pyproject.toml             # Project dependencies
└── uv.lock                    # Locked dependency versions
```
