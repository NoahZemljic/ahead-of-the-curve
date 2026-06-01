# Ahead of the Curve

> Identifying frontier AI models on the Hugging Face Hub _before_ they go mainstream.

**Ahead of the Curve** is an end-to-end MLOps system that surfaces high-potential models on the
Hugging Face Hub before they go mainstream, learning what a breakout release looks like in its
first hours and flagging the candidates most likely to take off.

---

## 1. Introduction

### Problem statement

Hugging Face has become the de facto platform for open-source AI models, with hundreds of new
models published every day. Deciding which models are worth the cost of evaluation,
fine-tuning or deployment before significant credibility has been established creates a real
operational challenge for companies and research institutions. A wrong bet not only costs time
and compute, but forces engineering teams to constantly scout for replacements and actively
undermines the productivity gains they were hoping these models would deliver.

This project builds a live ML system that makes that decision automatically. Given a model
uploaded in the last 24 hours, the system predicts both its expected download count and whether
it will land in the top quartile of download growth within 30 days. The range of topics is
deliberately narrow, focusing on three frontier niches where the cost of a wrong call is
greatest: **general-purpose robotics** (VLA models and embodied agents), **small language
models and edge inference**, and **multimodal reasoning**.

### Solution: Ahead of the Curve

Ahead of the Curve learns what promising early-stage models have in common and scores every new
release against that profile. For each candidate it predicts two targets:

- **`download_growth_30d`** — the model's expected 30-day download count (regression via XGBoost).
- **`top_quartile`** — whether it will land in the top 25% of its cohort by download growth
  (binary classification via XGBoost).

Predictions are refreshed every two hours and presented in a dashboard, ranked by the most
promising newcomers.

### Data source

All data comes from the **Hugging Face Hub**, queried through the `huggingface_hub` API:

- **Metadata** — downloads (30-day and all-time), likes, trending score, tags, library, and
  creation / last-modified timestamps.
- **Model cards** — the README text, which drives the semantic features.

The system focuses on three frontier topics: **robotics / VLA models**, **small language models
(SLMs)** and **multimodal reasoning models**.

### Features

From these raw signals the feature pipeline derives the inputs the models actually train on:

| Group                  | Features                                                                                       | Description                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Semantic relevance** | `relevance_robotics`, `relevance_slm`, `relevance_multimodal`, `best_topic`                    | Cosine similarity of the model-card embedding (using `all-MiniLM-L6-v2`) to each frontier-topic vector |
| **Temporal**           | `age_hours`                                                                                    | Model age in hours at snapshot time                                                                    |
| **Early momentum**     | `download_velocity_24h`, `download_velocity_72h`                                               | Downloads captured ~24h / ~72h after creation (imputed to 0 when unavailable)                          |
| **Metadata**           | `likes`, `trending_score`, `downloads_30d`, `downloads_all_time`, `tag_count`, `has_paper_tag` | Raw popularity and descriptive signals taken directly from the Hub                                     |

Before anything is stored or scored, a **relevance filter** drops models whose best topic score
falls below `0.25`, clearing out the long tail of off-topic uploads so the models only ever see
candidates worth ranking.

---

## 2. Project Structure

The system is built from a set of focused pipelines that hand off to one another: the feature
pipeline feeds the feature store, the training pipeline turns that data models
and the inference pipeline scores new releases for the dashboard to display. Each pipeline is
packaged independently with its own dependency group, Dockerfile and schedule so it can be
built and run in isolation.

### Components

| Component                | What it does                                                                                                                                                                                                       | Automation                                        | Frequency                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------- |
| **Feature pipeline**     | Fetches recently modified models, computes features, applies the relevance filter, labels mature (30+ day old) models, and writes feature rows to the Hopsworks feature store.                                     | GitHub Actions                                    | Daily, 22:00 UTC          |
| **Backfill pipeline**    | One-time historical population: pulls ~90 days of frontier-topic models so training has labelled data immediately, without waiting 30 days for live labels.                                                        | GitHub Actions, manual trigger                    | On demand                 |
| **Training pipeline**    | Loads labelled data, grid-searches XGBoost regressor + classifier, logs runs to the DagsHub MLflow registry, promotes a challenger only if it beats the `champion` by ≥ 1%, and deploys the promoted model to GCS. | GitHub Actions                                    | Weekly, Mon 02:00 UTC     |
| **Inference pipeline**   | Batch job that fetches models created in the last 2h, scores them with the GCS champion models, writes predictions to Hopsworks and `predictions/latest.json` on GCS.                                              | GCP Cloud Run **Job** + Cloud Scheduler           | Bi-hourly                 |
| **Inference deployment** | FastAPI **Service** that loads the champion models at startup and exposes `/predict`, `/predictions` and `/health`. This is what the dashboard talks to.                                                           | GCP Cloud Run **Service**                         | Always on (push-deployed) |
| **Dashboard**            | Gradio UI that reads predictions from the FastAPI service and presents them per topic, ranked by forecasted downloads.                                                                                             | Run locally (or deployed to a Hugging Face Space) | On demand                 |

---

## 3. Prerequisites

The following tools must be installed locally before setting the project up:

- **[Git](https://git-scm.com/downloads)**
- **[Python 3.11](https://www.python.org/downloads/)**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **[Docker](https://docs.docker.com/get-docker/)**
- **[Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install)**

---

## 4. Setup

The project relies on a few free external services, and all of their credentials live in a
single `.env` file. The steps below cover the local setup first, then each service in turn, filling in the corresponding environment variables as you go.

### 4.1 Clone the repository

```bash
git clone https://github.com/<your-org>/ahead-of-the-curve.git
cd ahead-of-the-curve
```

### 4.2 Install dependencies with `uv`

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/), and Python is pinned to 3.11
(see `.python-version`). Install everything in one go, or just the group you need:

```bash
# Everything (all pipeline groups + dev)
uv sync --all-groups

# …or a single pipeline's dependencies:
uv sync --group feature      # feature + backfill pipeline
uv sync --group training     # training pipeline
uv sync --group inference    # FastAPI inference service
uv sync --group dashboard    # Gradio dashboard
```

> On Apple Silicon, Docker images must be built with `--platform linux/amd64`, as the
> `hops-deltalake` dependency has no arm64 wheel.

### 4.3 Create your environment file

Every credential lives in the `.env` file. Start from the template:

```bash
cp .env.example .env
```

Below is the full set of variables. You don't need values yet — the following sections walk
through obtaining each one, service by service.

| Variable                                                               | Role                                                                                        |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `HF_TOKEN`                                                             | Authenticates requests to the Hugging Face Hub API                                          |
| `HOPSWORKS_API_KEY`                                                    | Authenticates to the Hopsworks feature store                                                |
| `DAGSHUB_USER_TOKEN`, `DAGSHUB_REPO_OWNER`, `DAGSHUB_REPO_NAME`        | Authenticate and address the DagsHub-hosted MLflow tracking server and model registry       |
| `GCP_PROJECT_ID`, `GCP_REGION`, `GCS_BUCKET_NAME`, `CLOUD_RUN_SERVICE` | Identify the GCP project, region, model/prediction bucket and Cloud Run service to deploy   |
| `GOOGLE_APPLICATION_CREDENTIALS`                                       | Path to the service-account JSON used for GCP authentication                                |
| `INFERENCE_API_URL`, `PORT`                                            | Base URL of the inference API the dashboard reads from, and the port the dashboard binds to |

### 4.4 Hugging Face

The Hub is where all model data comes from. Follow the following steps below to set it up:

1. Sign up at [huggingface.co/join](https://huggingface.co/join) with your email and a password,
   then verify your address.
2. Click your avatar (top right) → **Settings** → **Access Tokens** → **Create new token**.
3. Choose a **Fine-grained** token and grant only the **Read** repositories permission. Name it (e.g. `ahead-of-the-curve`) and click **Create token**.
4. Copy the token into `.env`:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

### 4.5 Hopsworks

Hopsworks holds the engineered features and the predictions.

1. Sign up at [app.hopsworks.ai](https://app.hopsworks.ai) (Google, GitHub, or email).
2. On the landing page, click **Create New Project**, name it and confirm.
3. Open the account menu (top right) → **Account Settings** → **API** → **New API key**.
   Name it, ensure the `featurestore`, `project`, and `job` scopes are selected, and create it.
4. Copy the key into `.env`:

   ```env
   HOPSWORKS_API_KEY=your_hopsworks_api_key
   ```

   Note: The feature groups (`frontier_models_features` and
   `frontier_models_predictions`) are created automatically on first write.

### 4.6 DagsHub

DagsHub provides the hosted MLflow server where the training pipeline logs runs and keeps thechampion model.

1. Sign up at [dagshub.com](https://dagshub.com) (GitHub, Google, or email).
2. Click **+ → New Repository** → Select **Create blank repository** → Name it (e.g. `ahead-of-the-curve`) and choose its visibility → **Create Repository**. DagsHub provisions a hosted **MLflow** tracking server and model
   registry behind it automatically — the training pipeline connects through `dagshub.init(...)`,
   so no further MLflow configuration is required.

3. Get a token from your avatar → **Settings** → **Tokens** (the default user token works, or
   create a new one).
4. Fill in the DagsHub variables — `DAGSHUB_REPO_OWNER` is your username and `DAGSHUB_REPO_NAME`
   is the repo you just created:
   ```env
   DAGSHUB_USER_TOKEN=your_dagshub_user_token
   DAGSHUB_REPO_OWNER=your_dagshub_username
   DAGSHUB_REPO_NAME=ahead-of-the-curve
   ```

### 4.7 Google Cloud

Google Cloud stores the trained models and runs inference. The whole setup is done through the
Cloud Console UI:

1. **Create a project.**
   - Log into the [Google Cloud Console](https://console.cloud.google.com).
   - Click the project dropdown at the top of the screen and select **New Project**.
   - **Project Name:** you can name this anything (e.g. `ahead-of-the-curve`).
   - **Project ID:** Google auto-generates an ID under the name. This ID must be globally unique
     across all of Google Cloud. You can edit it to something like `ahead-of-the-curve-1`.
     Note this ID down.
   - Click **Create**.
2. **Enable the required APIs.** Under **APIs & Services → Library**, search for and **Enable**
   each of the following:
   - Cloud Run Admin API
   - Cloud Scheduler API
   - Artifact Registry API
   - Cloud Storage API
3. **Create a storage bucket.** Go to **Cloud Storage → Buckets → Create**, give it a globally
   unique name, choose your region (`europe-west6`) and keep the defaults. This is where
   the pipelines write the champion models and predictions.
4. **Create an Artifact Registry repository.** Under **Artifact Registry → Repositories → Create
   Repository**. Name it `inference`, set the format to **Docker**, pick the same region and
   create it. The inference images are pushed here.
5. **Create a service account.** Go to **IAM & Admin → Service Accounts → Create Service
   Account**, name it (e.g. `ahead-of-the-curve`) and click **Create and Continue**. On the
   "Grant this service account access" step, add the roles below. They are scoped to the minimum
   the project actually uses .

   | Role                                                           | Why it is needed                                      |
   | -------------------------------------------------------------- | ----------------------------------------------------- |
   | **Cloud Run Admin** (`roles/run.admin`)                        | Deploys the inference Service and Job                 |
   | **Cloud Run Invoker** (`roles/run.invoker`)                    | Lets Cloud Scheduler trigger the batch Job            |
   | **Service Account User** (`roles/iam.serviceAccountUser`)      | Acts as the runtime SA when deploying Cloud Run       |
   | **Artifact Registry Writer** (`roles/artifactregistry.writer`) | Pushes inference Docker images                        |
   | **Cloud Scheduler Admin** (`roles/cloudscheduler.admin`)       | Creates/updates the bi-hourly schedule                |
   | **Storage Object Admin** (`roles/storage.objectAdmin`)         | Reads/writes model + prediction objects in the bucket |

   Then click **Done**.

6. **Download the key.** Open the new service account → **Keys → Add Key → Create new key →
   JSON → Create**, and save the file as `gcp-key.json` inside the repo's
   `.secrets/` directory.
7. **Fill in the remaining variables** in `.env`:
   ```env
   GCP_PROJECT_ID=your_gcp_id
   GCP_REGION=europe-west6
   GCS_BUCKET_NAME=your-bucket-name
   GOOGLE_APPLICATION_CREDENTIALS=.secrets/gcp-key.json
   CLOUD_RUN_SERVICE=inference-pipeline
   ```

Your `.env` is now complete and the project is ready to run.

> In CI, these same values are provided as GitHub Actions repository **secrets**.

---

## 5. Running the Pipelines

Every pipeline runs locally with `uv`, and the production ones also deploy to GCP through the
workflows in `.github/workflows/`. The order below follows the flow of data: populate the
store, train on it, then serve predictions. This work well as a first end-to-end run.

### 5.1 Backfill (run once, first)

A new feature store is empty and live labels take 30 days to mature. The backfill gives
training data to learn from immediately by pulling ~90 days of frontier models in one pass:

```bash
uv run python -m src.feature_pipeline.backfill_pipeline
```

In GitHub Actions, trigger the **Feature Pipeline Backfill** workflow manually using `workflow_dispatch`.

### 5.2 Feature pipeline (daily)

The feature pipeline keeps the store current, fetching each day's new models and appending their
features:

```bash
uv run python -m src.feature_pipeline.pipeline
```

In Github Actions, it runs automatically every day at **22:00 UTC** via `feature-pipeline.yml`.

### 5.3 Training pipeline (weekly)

Trains both models, logs every run to DagsHub MLflow and promotes a challenger to
`champion` only when it beats the incumbent by at least 1%. The promoted model is
uploaded to GCS as a `model.joblib`, which in turn triggers a fresh revision of the inference service:

```bash
uv run python -m src.training_pipeline.pipeline
```

Runs weekly, every **Monday at 02:00 UTC**, via `training-pipeline.yml`.

### 5.4 Inference pipeline (bi-hourly)

The inference pipeline is the batch job that scores newly uploaded models against the current
champions. To run it locally:

```bash
uv run python -m src.inference_pipeline.pipeline
```

To deploy it, push a change under `src/inference_pipeline/` (or trigger `inference-pipeline.yml`
manually). The workflow builds `docker/inference-pipeline`, deploys it as a Cloud Run **Job**
and creates a Cloud Scheduler trigger that fires it **every two hours** (`0 */2 * * *`).

To build and deploy by hand, run from the project root with `-f`:

```bash
IMAGE=<region>-docker.pkg.dev/<project>/inference/inference-pipeline:latest

docker build --platform linux/amd64 -f docker/inference-pipeline/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

gcloud run jobs deploy inference-pipeline \
  --image="$IMAGE" --region=<region> --memory=4Gi --cpu=2 \
  --set-env-vars=HF_TOKEN=...,HOPSWORKS_API_KEY=...,GCS_BUCKET_NAME=...
```

### 5.5 Inference deployment

The predictions are served through a FastAPI service It deploys as a Cloud Run **Service** whenever you push under `src/inference_pipeline/` or
`docker/inference-deployment/` (or trigger `inference-deployment.yml`). For local development:

```bash
uv run uvicorn src.inference_pipeline.main:app --reload --port 8080
# → GET  /health
#   GET  /predictions?limit=50
#   POST /predict
```

### 5.6 Dashboard (run locally)

Point the dashboard at the API (your deployed service, or the local one above) and launch it:

```env
# in .env
INFERENCE_API_URL=https://<your-cloud-run-service-url>   # or http://0.0.0.0:8080 for local
PORT=7860
```

```bash
uv run python src/dashboard/app.py
# → http://0.0.0.0:7860
```

The result is the full picture: frontier releases ranked by their forecasted 30-day downloads.
The dashboard is also run as a [**Hugging Face Space**](https://huggingface.co/spaces/NoahZemljic/ahead-of-the-curve) for demo purposes.

---

## 6. Repository Layout

A map of the repository, with each directory annotated by the role it plays:

```
ahead-of-the-curve/
├── .github/                            # GitHub Actions workflows
├── .secrets/                           # GCP service-account keys
├── docker/                             # Dockerfiles (one per pipeline)
├── documents/                          # Project proposal
├── notebooks/                          # EDA / prototyping
├── src/
│   ├── common/                         # shared Hopsworks client base class
│   ├── dashboard/                      # Gradio UI
│   ├── feature_pipeline/               # ingest → features → labels → feature store (+ backfill)
│   ├── inference_pipeline/             # batch scoring job + FastAPI serving service
│   └── training_pipeline/              # load → train → promote champion → deploy to GCS
├── .env.example                        # Environment variable template
├── .gitignore                          # Git ignore rules
├── .python-version                     # Python version pin (3.11)
├── README.md                           # you are reading this
├── pyproject.toml                      # dependencies (per-pipeline groups)
└── uv.lock                             # locked dependency versions
```
