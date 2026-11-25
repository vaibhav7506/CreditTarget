"""Pydantic v2 application settings and deep-merge configuration loader."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from credittarget.exceptions import ConfigError


class StorageConfig(BaseModel):
    backend: Literal["local", "s3"] = "local"
    base_dir: str = "artifacts"
    delta_dir: str = "artifacts/delta"
    s3_bucket: str = ""
    s3_prefix: str = ""


class SparkConfig(BaseModel):
    app_name: str = "CreditTarget"
    master: str = "local[2]"
    shuffle_partitions: int = 4
    driver_memory: str = "2g"


class DataGenConfig(BaseModel):
    num_customers: int = 2000
    num_months: int = 12
    seed: int = 42


class PolicyConfig(BaseModel):
    lgd: float = 0.45
    default_promo_cost: float = 15.0
    max_pd_cutoff: float = 0.08
    top_k_budget_fraction: float = 0.30


class SplitsConfig(BaseModel):
    train_months: list[str] = Field(
        default_factory=lambda: [
            "2025-01",
            "2025-02",
            "2025-03",
            "2025-04",
            "2025-05",
            "2025-06",
            "2025-07",
            "2025-08",
        ]
    )
    val_months: list[str] = Field(default_factory=lambda: ["2025-09"])
    oot_months: list[str] = Field(default_factory=lambda: ["2025-10", "2025-11", "2025-12"])


class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    forbidden_leakage: list[str] = Field(default_factory=list)


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///artifacts/mlflow/mlflow.db"
    experiment_name: str = "credittarget"
    propensity_model_name: str = "credittarget_propensity"
    pd_model_name: str = "credittarget_pd"


class GovernanceThresholds(BaseModel):
    max_psi_critical: float = 0.25
    max_psi_warning: float = 0.10
    max_ks_drop_pct: float = 15.0
    min_oot_gini_propensity: float = 0.35
    min_oot_gini_pd: float = 0.40
    min_top2_decile_lift: float = 1.80
    max_brier_score: float = 0.20
    zero_leakage_violations: bool = True
    min_disparate_impact_ratio: float = 0.80


class AppConfig(BaseModel):
    app_name: str = "CreditTarget"
    seed: int = 42
    storage: StorageConfig = Field(default_factory=StorageConfig)
    spark: SparkConfig = Field(default_factory=SparkConfig)
    datagen: DataGenConfig = Field(default_factory=DataGenConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    governance: GovernanceThresholds = Field(default_factory=GovernanceThresholds)


def _deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges dict2 into dict1."""
    merged = copy.deepcopy(dict1)
    for k, v in dict2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {p.resolve()}")
    try:
        with open(p, encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML file {p}: {e}") from e


def load_config(
    profile_path: str | Path | None = None,
    base_path: str | Path = "configs/base.yaml",
    governance_path: str | Path = "configs/governance_policy.yaml",
) -> AppConfig:
    """Load base, profile, governance configs, and environment variable overrides."""
    # 1. Base YAML
    merged: dict[str, Any] = {}
    if Path(base_path).exists():
        merged = _load_yaml(base_path)

    # 2. Profile YAML (e.g. local.yaml, benchmark.yaml, cloud_aws.yaml)
    if profile_path:
        profile_data = _load_yaml(profile_path)
        merged = _deep_merge(merged, profile_data)

    # 3. Governance Policy YAML
    if Path(governance_path).exists():
        gov_data = _load_yaml(governance_path)
        merged = _deep_merge(merged, gov_data)

    # 4. Environment variable overrides (e.g., STORAGE_BACKEND -> storage.backend)
    storage_backend = os.getenv("STORAGE_BACKEND") or os.getenv("CREDITTARGET_STORAGE_BACKEND")
    if storage_backend:
        if "storage" not in merged:
            merged["storage"] = {}
        merged["storage"]["backend"] = storage_backend

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("CREDITTARGET_MLFLOW_TRACKING_URI")
    if mlflow_uri:
        if "mlflow" not in merged:
            merged["mlflow"] = {}
        merged["mlflow"]["tracking_uri"] = mlflow_uri

    try:
        return AppConfig.model_validate(merged)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e
