"""
MLflow Model Registry Service

Provides model registration, versioning, and retrieval from MLflow.
When MLFLOW_ENABLED=false, all functions gracefully no-op.
"""
import torch
from pathlib import Path
from loguru import logger
from app.config import settings


def _get_client():
    """Lazily import and configure the MLflow client."""
    if not settings.mlflow_enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        return mlflow
    except ImportError:
        logger.warning("mlflow package not installed — registry disabled")
        return None


def register_model(
    model_name: str,
    weights_path: str,
    metrics: dict | None = None,
    params: dict | None = None,
    tags: dict | None = None,
) -> str | None:
    """
    Log a trained model to MLflow with metrics and params.
    Returns the run_id or None if MLflow is disabled.
    """
    mlflow = _get_client()
    if mlflow is None:
        return None

    with mlflow.start_run() as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        # Log the weights file as an artifact
        mlflow.log_artifact(weights_path, artifact_path="weights")

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Register model version
        model_uri = f"runs:/{run.info.run_id}/weights"
        mlflow.register_model(model_uri, model_name)

        logger.info(f"Registered model '{model_name}' — run_id={run.info.run_id}")
        return run.info.run_id


def load_production_weights(model_name: str) -> str | None:
    """
    Fetch the latest 'Production' model weights path from MLflow.
    Returns the local path to the downloaded artifact, or None.
    """
    mlflow = _get_client()
    if mlflow is None:
        return None

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        # Get latest version with 'Production' alias
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning(f"No 'Production' version found for '{model_name}'")
            return None

        latest = versions[0]
        artifact_uri = latest.source
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        logger.info(f"Loaded production weights for '{model_name}' v{latest.version}")
        return local_path

    except Exception as e:
        logger.error(f"Failed to load production weights for '{model_name}': {e}")
        return None


def load_shadow_weights(model_name: str) -> str | None:
    """
    Fetch the latest 'Staging' model weights path from MLflow.
    Used for shadow deployment comparisons.
    """
    mlflow = _get_client()
    if mlflow is None:
        return None

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not versions:
            return None

        latest = versions[0]
        artifact_uri = latest.source
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        logger.info(f"Loaded shadow weights for '{model_name}' v{latest.version}")
        return local_path

    except Exception as e:
        logger.warning(f"Failed to load shadow weights for '{model_name}': {e}")
        return None
