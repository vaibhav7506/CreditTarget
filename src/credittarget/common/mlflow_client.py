"""MLflow model registry client with lineage tagging, alias management, and artifact loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient

from credittarget.config import MLflowConfig
from credittarget.exceptions import ModelNotFoundError
from credittarget.logging_utils import get_logger

logger = get_logger("ModelRegistryClient")


class ModelRegistryClient:
    """Encapsulates all MLflow interactions: runs, tags, artifacts, aliases, and model loading."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._setup_tracking_uri()
        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def _setup_tracking_uri(self) -> None:
        uri = self.config.tracking_uri
        if uri.startswith("sqlite:///"):
            db_path = uri.replace("sqlite:///", "")
            # Windows path handling for sqlite:///C:/... or sqlite:///artifacts/...
            p = Path(db_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)

    def start_run(self, run_name: str | None = None, tags: dict[str, Any] | None = None) -> Any:
        """Starts an active MLflow run."""
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_lineage_tags(
        self,
        git_sha: str,
        gold_delta_version: int,
        feature_schema_hash: str,
        config_hash: str,
        additional_tags: dict[str, Any] | None = None,
    ) -> None:
        """Logs standard model risk lineage tags to the current active run."""
        tags: dict[str, Any] = {
            "credittarget.git_sha": git_sha,
            "credittarget.gold_delta_version": str(gold_delta_version),
            "credittarget.feature_schema_hash": feature_schema_hash,
            "credittarget.config_hash": config_hash,
        }
        if additional_tags:
            tags.update(additional_tags)
        mlflow.set_tags(tags)

    def log_model_and_register(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str,
        signature: ModelSignature | None = None,
        input_example: Any = None,
        initial_alias: str = "challenger",
    ) -> tuple[str, str]:
        """Logs a model artifact, registers it in the Model Registry, and assigns initial alias.

        Returns (model_uri, model_version).
        """
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        # Retrieve registered model version
        mv = self.client.get_latest_versions(registered_model_name)[0]
        model_version = str(mv.version)

        # Set initial alias (e.g. challenger)
        self.set_model_alias(registered_model_name, initial_alias, model_version)
        logger.info(
            "model_registered",
            model_name=registered_model_name,
            version=model_version,
            alias=initial_alias,
        )
        return model_info.model_uri, model_version

    def set_model_alias(self, model_name: str, alias: str, version: str) -> None:
        """Assigns an alias (@champion, @challenger) to a specific model version."""
        self.client.set_registered_model_alias(model_name, alias, version)

    def promote_to_champion(self, model_name: str, version: str) -> None:
        """Promotes a model version to @champion."""
        self.set_model_alias(model_name, "champion", version)
        logger.info("model_promoted_to_champion", model_name=model_name, version=version)

    def load_model_by_alias(self, model_name: str, alias: str = "champion") -> Any:
        """Loads a model directly using its alias (e.g., models:/credittarget_propensity@champion)."""
        model_uri = f"models:/{model_name}@{alias}"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to load model '{model_name}' with alias '@{alias}': {e}"
            ) from e

    def get_model_version_by_alias(self, model_name: str, alias: str = "champion") -> str:
        """Retrieves the model version string associated with an alias."""
        try:
            model_version = self.client.get_model_version_by_alias(model_name, alias)
            return str(model_version.version)
        except Exception as e:
            raise ModelNotFoundError(
                f"Alias '@{alias}' not found on registered model '{model_name}': {e}"
            ) from e
