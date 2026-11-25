"""Lineage manifest builder aggregating Git SHA, Delta versions, schema hashes, and model IDs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from credittarget.common.artifact_store import ArtifactStore
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig
from credittarget.modeling.train import get_git_sha


def build_lineage_manifest(
    config: AppConfig,
    artifact_store: ArtifactStore,
    delta_manager: DeltaTableManager,
    mlflow_client: ModelRegistryClient,
) -> dict[str, Any]:
    """Assembles a comprehensive audit manifest linking data, schema, code, and model versions."""
    gold_path = f"{config.storage.delta_dir}/gold_customer_features"
    gold_version = delta_manager.get_latest_version(gold_path)

    prop_version = "unknown"
    pd_version = "unknown"
    try:
        prop_version = mlflow_client.get_model_version_by_alias(
            config.mlflow.propensity_model_name, "champion"
        )
    except Exception:
        try:
            prop_version = mlflow_client.get_model_version_by_alias(
                config.mlflow.propensity_model_name, "challenger"
            )
        except Exception:
            pass

    try:
        pd_version = mlflow_client.get_model_version_by_alias(
            config.mlflow.pd_model_name, "champion"
        )
    except Exception:
        try:
            pd_version = mlflow_client.get_model_version_by_alias(
                config.mlflow.pd_model_name, "challenger"
            )
        except Exception:
            pass

    git_sha = get_git_sha()

    manifest: dict[str, Any] = {
        "platform": "CreditTarget",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "gold_delta_version": gold_version,
        "propensity_model_name": config.mlflow.propensity_model_name,
        "propensity_model_version": prop_version,
        "pd_model_name": config.mlflow.pd_model_name,
        "pd_model_version": pd_version,
        "storage_backend": config.storage.backend,
        "app_config": config.model_dump(),
    }

    artifact_store.write_json("governance/lineage.json", manifest)
    return manifest
