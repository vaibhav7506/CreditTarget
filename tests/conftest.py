"""Pytest fixtures for CreditTarget testing suite."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from credittarget.common.artifact_store import ArtifactStore
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig, load_config
from credittarget.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Session-scoped SparkSession with Delta Lake enabled."""
    if "JAVA_HOME" not in os.environ or not os.path.isdir(os.environ["JAVA_HOME"]):
        adoptium = "C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.21.10002-hotspot"
        if os.path.isdir(adoptium):
            os.environ["JAVA_HOME"] = adoptium
            os.environ["PATH"] = f"{adoptium}\\bin;{os.environ.get('PATH', '')}"

    if "HADOOP_HOME" not in os.environ or not os.path.isdir(os.environ["HADOOP_HOME"]):
        hadoop_dir = "C:\\hadoop"
        if os.path.isdir(hadoop_dir):
            os.environ["HADOOP_HOME"] = hadoop_dir
            os.environ["PATH"] = f"{hadoop_dir}\\bin;{os.environ.get('PATH', '')}"

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    test_config = load_config("configs/local.yaml")
    test_config.spark.shuffle_partitions = 2
    session = get_spark_session(test_config)
    yield session


@pytest.fixture(scope="session")
def tmp_workspace() -> Generator[Path, None, None]:
    """Temporary test directory cleaned up after session."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="credittarget_test_"))
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def app_config(tmp_workspace: Path) -> AppConfig:
    """AppConfig configured to use isolated temporary paths."""
    config = load_config("configs/local.yaml")
    config.storage.base_dir = str(tmp_workspace / "artifacts")
    config.storage.delta_dir = str(tmp_workspace / "artifacts" / "delta")
    config.mlflow.tracking_uri = f"sqlite:///{(tmp_workspace / 'mlflow.db').as_posix()}"
    config.datagen.num_customers = 60
    return config


@pytest.fixture
def artifact_store(app_config: AppConfig) -> ArtifactStore:
    """ArtifactStore instance configured for temporary test storage."""
    return ArtifactStore(app_config.storage)


@pytest.fixture
def delta_manager(spark: SparkSession) -> DeltaTableManager:
    """DeltaTableManager instance."""
    return DeltaTableManager(spark)


@pytest.fixture
def mlflow_client(app_config: AppConfig) -> ModelRegistryClient:
    """ModelRegistryClient instance with isolated SQLite backend."""
    return ModelRegistryClient(app_config.mlflow)
