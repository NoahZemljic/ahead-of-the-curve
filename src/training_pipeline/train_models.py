import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import dagshub

logger = logging.getLogger(__name__)


class Trainer:
    """Train XGBoost regression and classification models with GridSearchCV and Pipeline."""

    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        """Initialize training, evaluation, MLflow, and model registry settings.

        Args:
            test_size: Fraction of rows reserved for final model evaluation.
            val_size: Fraction of rows added back into the final training fit after tuning.
            random_state: Seed used for reproducible train/validation/test splits.
        """
        self.REGRESSION_MODEL_NAME = "regressor"
        self.CLASSIFICATION_MODEL_NAME = "classifier"
        self.REGRESSOR_REGISTRY_NAME = "champion-regressor"
        self.CLASSIFIER_REGISTRY_NAME = "champion-classifier"
        self.DAGSHUB_REPO_OWNER = "NoahZemljic"
        self.DAGSHUB_REPO_NAME = "ahead-of-the-curve"
        self.CHAMPION_ALIAS = "champion"
        self.PROMOTION_THRESHOLD = 0.01
        self.N_FOLDS = 10
        self.params = {
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 200],
        }
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.configure_mlflow_tracking()

    def configure_mlflow_tracking(self):
        """Configure MLflow to use the DagsHub-hosted tracking server and registry."""
        load_dotenv()

        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", self.DAGSHUB_REPO_OWNER)
        repo_name = os.getenv("DAGSHUB_REPO_NAME", self.DAGSHUB_REPO_NAME)
        repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        dagshub_username = os.getenv("DAGSHUB_USERNAME", repo_owner)

        if dagshub_token:
            os.environ.setdefault("DAGSHUB_USER_TOKEN", dagshub_token)
            os.environ.setdefault("MLFLOW_TRACKING_USERNAME", dagshub_username)
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", dagshub_token)

        dagshub.init(url=repo_url, root=os.getcwd(), mlflow=True)

        # dagshub.init configures the tracking URI; the registry URI is set explicitly
        # because this pipeline registers and promotes models after logging them.
        mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])

    def build_pipeline(self, model_type, numeric_cols):
        """Build a preprocessing and XGBoost pipeline for one prediction task."""
        if model_type == "regressor":
            model = XGBRegressor(eval_metric="rmse")
        else:
            model = XGBClassifier(eval_metric="logloss")

        scaler = ColumnTransformer(
            [("scale", StandardScaler(), numeric_cols)],
            remainder="passthrough",
        )

        return Pipeline([
            ("scaler", scaler),
            ("model", model),
        ])

    def cross_validate(self, splits, model_type):
        """Tune hyperparameters and refit the best model on train + validation data.

        Uses GridSearchCV with only the training split for cross-validated model selection.
        After the best parameters are found, the final model is retrained on the combined
        training and validation splits.
        """
        numeric_cols = splits["numeric_cols"]
        pipeline = self.build_pipeline(model_type, numeric_cols)

        if model_type == "regressor":
            scoring_metric = "neg_root_mean_squared_error"
            y_train = splits["y_reg_train"]
            y_val = splits["y_reg_val"]
        else:
            scoring_metric = "f1"
            y_train = splits["y_clf_train"]
            y_val = splits["y_clf_val"]

        # Tune parameters only on the training split
        grid_search = GridSearchCV(
            pipeline,
            self.params,
            cv=self.N_FOLDS,
            scoring=scoring_metric,
            n_jobs=-1,
        )
        grid_search.fit(splits["X_train"], y_train)

        mlflow.log_params({
            f"{model_type}.{k.replace('model__', '')}": v
            for k, v in grid_search.best_params_.items()
        })
        logger.info(f"{model_type} best params: {grid_search.best_params_}")
        logger.info(f"{model_type} best CV score: {grid_search.best_score_:.4f}")

        best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}

        X_train_val = pd.concat([splits["X_train"], splits["X_val"]])
        y_train_val = pd.concat([y_train, y_val])

        # Refit preprocessing and model together on training and validation sets
        final_pipeline = self.build_pipeline(model_type, numeric_cols)
        final_pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
        final_pipeline.fit(X_train_val, y_train_val)

        final_model = final_pipeline.named_steps["model"]
        scaler = final_pipeline.named_steps["scaler"]

        return final_model, scaler

    def evaluate_models(self, regressor, classifier, splits, scaler):
        """Evaluate both trained models on the test split and log metrics."""
        X_test_scaled = scaler.transform(splits["X_test"])

        reg_preds = regressor.predict(X_test_scaled)
        clf_preds = classifier.predict(X_test_scaled)

        metrics = {
            "test_rmse": root_mean_squared_error(splits["y_reg_test"], reg_preds),
            "test_mae": mean_absolute_error(splits["y_reg_test"], reg_preds),
            "test_r2": r2_score(splits["y_reg_test"], reg_preds),
            "test_accuracy": accuracy_score(splits["y_clf_test"], clf_preds),
            "test_f1": f1_score(splits["y_clf_test"], clf_preds),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Test metrics: {metrics}")
        return metrics

    def promote_model(self, model_uri, metrics, model_type):
        """Register and promote a model when it beats the current champion alias.

        The candidate is compared against the champion model alias in the DagsHub
        MLflow registry using test RMSE for the regressor and test F1 for the classifier.
        """
        client = MlflowClient()

        if model_type == "regressor":
            model_name = self.REGRESSOR_REGISTRY_NAME
            metric_name = "test_rmse"
            candidate_metric = metrics["test_rmse"]
            lower_is_better = True
        else:
            model_name = self.CLASSIFIER_REGISTRY_NAME
            metric_name = "test_f1"
            candidate_metric = metrics["test_f1"]
            lower_is_better = False

        champion_metric = None
        try:
            champion_version = client.get_model_version_by_alias(
                model_name, self.CHAMPION_ALIAS
            )
            champion_run = client.get_run(champion_version.run_id)
            champion_metric = champion_run.data.metrics.get(metric_name)
        except Exception:
            # A missing registry/model/alias means this is the first promotable candidate
            pass

        if champion_metric is not None:
            if lower_is_better:
                is_better = candidate_metric < champion_metric * (1 - self.PROMOTION_THRESHOLD)
            else:
                is_better = candidate_metric > champion_metric * (1 + self.PROMOTION_THRESHOLD)

            if not is_better:
                logger.info(
                    f"{model_name}: candidate {metric_name}={candidate_metric:.4f} not better than "
                    f"champion {metric_name}={champion_metric:.4f}, skipping promotion"
                )
                return

        mv = mlflow.register_model(model_uri, model_name)
        client.set_registered_model_alias(model_name, self.CHAMPION_ALIAS, mv.version)
        logger.info(
            f"{model_name} v{mv.version} promoted to {self.CHAMPION_ALIAS} "
            f"({metric_name}={candidate_metric:.4f})"
        )

    def split_data(self, data: dict) -> dict:
        """Separate targets from features and split data into train/validation/test sets."""
        df = data["df"].copy()

        y_reg = df["download_growth_30d"]
        y_clf = df["top_quartile"]
        X = df.drop(columns=["download_growth_30d", "top_quartile"])

        X_train, X_temp, y_reg_train, y_reg_temp, y_clf_train, y_clf_temp = train_test_split(
            X, y_reg, y_clf,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
        )

        relative_val_size = self.val_size / (self.test_size + self.val_size)

        X_val, X_test, y_reg_val, y_reg_test, y_clf_val, y_clf_test = train_test_split(
            X_temp, y_reg_temp, y_clf_temp,
            test_size=1 - relative_val_size,
            random_state=self.random_state,
        )

        logger.info(
            f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_reg_train": y_reg_train, "y_reg_val": y_reg_val, "y_reg_test": y_reg_test,
            "y_clf_train": y_clf_train, "y_clf_val": y_clf_val, "y_clf_test": y_clf_test,
            "numeric_cols": data["numeric_cols"],
        }

    def train(self, data):
        """Run the full training workflow and promote eligible models."""
        splits = self.split_data(data)

        with mlflow.start_run():
            # Train the regression and classification candidates
            regressor, reg_scaler = self.cross_validate(splits, model_type="regressor")
            classifier, _ = self.cross_validate(splits, model_type="classifier")

            metrics = self.evaluate_models(regressor, classifier, splits, reg_scaler)

            # Registerable models must be logged before MLflow model promotion.
            regressor_info = mlflow.xgboost.log_model(
                regressor, name=self.REGRESSION_MODEL_NAME
            )
            classifier_info = mlflow.xgboost.log_model(
                classifier, name=self.CLASSIFICATION_MODEL_NAME
            )

            self.promote_model(regressor_info.model_uri, metrics, model_type="regressor")
            self.promote_model(classifier_info.model_uri, metrics, model_type="classifier")
