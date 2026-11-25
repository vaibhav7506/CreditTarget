"""Distributed PySpark batch scoring stage with broadcast driver-loaded models and idempotent Delta writes."""

from __future__ import annotations

from typing import Any

try:
    import lightgbm as _lgb  # noqa: F401 - Pin LightGBM C++ runtime bindings before MLflow
except (ImportError, OSError):
    pass
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from credittarget.common.base_stage import BasePipelineStage, StageResult
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig
from credittarget.modeling.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from credittarget.modeling.policy import TargetingPolicyEngine
from credittarget.schemas.contracts import compute_schema_hash


def _score_worker(
    input_parquet: str,
    output_parquet: str,
    config_json: str,
) -> None:
    """Isolated child worker process for LightGBM/policy scoring without JVM collision."""
    import json

    import pandas as pd

    with open(config_json, encoding="utf-8") as f:
        config = AppConfig.model_validate(json.load(f))

    mlflow_client = ModelRegistryClient(config.mlflow)
    policy_engine = TargetingPolicyEngine(config)

    prop_model = mlflow_client.load_model_by_alias(config.mlflow.propensity_model_name, "champion")
    pd_model = mlflow_client.load_model_by_alias(config.mlflow.pd_model_name, "champion")
    prop_version = mlflow_client.get_model_version_by_alias(
        config.mlflow.propensity_model_name, "champion"
    )
    pd_version = mlflow_client.get_model_version_by_alias(config.mlflow.pd_model_name, "champion")

    pdf = pd.read_parquet(input_parquet)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pdf[feature_cols]

    pdf["propensity_score"] = prop_model.predict_proba(X)[:, 1]
    pdf["pd_score"] = pd_model.predict_proba(X)[:, 1]

    scored_pdf = policy_engine.evaluate_policy_pandas(pdf)
    scored_pdf["propensity_model_version"] = prop_version
    scored_pdf["pd_model_version"] = pd_version

    scored_pdf.to_parquet(output_parquet, index=False)


class BatchScoreStage(BasePipelineStage):
    """Driver-loads champion models, broadcasts to Spark workers, and scores target snapshot month."""

    def __init__(self, spark: SparkSession, config: AppConfig) -> None:
        super().__init__(config=config, stage_name="BatchScoreStage")
        self.spark = spark
        self.delta_manager = DeltaTableManager(spark)
        self.mlflow_client = ModelRegistryClient(config.mlflow)
        self.policy_engine = TargetingPolicyEngine(config)

    def extract(self, *args: Any, **kwargs: Any) -> DataFrame:
        """Extracts feature rows for the target snapshot month from gold."""
        snapshot_month = kwargs.get("snapshot_month", "2025-12")
        gold_path = f"{self.config.storage.delta_dir}/gold_customer_features"
        gold_df = self.delta_manager.read_table(gold_path)
        return gold_df.filter(F.col("snapshot_month") == snapshot_month)

    def transform(self, data: DataFrame, *args: Any, **kwargs: Any) -> DataFrame:
        """Driver loads champion models, executes vectorized scoring, and applies suppression rules."""
        import json
        import os
        import subprocess
        import sys
        import tempfile

        import pandas as pd

        pdf = pd.DataFrame(data.toPandas())
        gold_path = f"{self.config.storage.delta_dir}/gold_customer_features"
        gold_version = self.delta_manager.get_latest_version(gold_path)
        schema_hash = compute_schema_hash(data.schema)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_parquet = os.path.join(tmp_dir, "input.parquet")
            output_parquet = os.path.join(tmp_dir, "output.parquet")
            config_json = os.path.join(tmp_dir, "config.json")

            pdf.to_parquet(input_parquet, index=False)
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f)

            cmd = [
                sys.executable,
                "-m",
                "credittarget.serving.batch_score",
                "--input-parquet",
                input_parquet,
                "--output-parquet",
                output_parquet,
                "--config-json",
                config_json,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Isolated batch scoring failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                )

            scored_pdf = pd.read_parquet(output_parquet)

        scored_pdf["gold_delta_version"] = gold_version
        scored_pdf["feature_schema_hash"] = schema_hash

        result_cols = [
            "customer_id",
            "snapshot_month",
            "propensity_score",
            "pd_score",
            "expected_net_value",
            "target_decision",
            "suppression_reason",
            "propensity_model_version",
            "pd_model_version",
            "gold_delta_version",
            "feature_schema_hash",
        ]
        spark_scored = self.spark.createDataFrame(scored_pdf[result_cols]).withColumn(
            "scored_at_utc", F.current_timestamp()
        )
        return spark_scored

    def validate(self, data: DataFrame, *args: Any, **kwargs: Any) -> None:
        """Assert scored outputs are non-empty."""
        assert data.count() > 0, "Batch scoring produced 0 records."

    def load(self, data: DataFrame, *args: Any, **kwargs: Any) -> StageResult:
        """Idempotently writes scored partition using Delta replaceWhere."""
        snapshot_month = kwargs.get("snapshot_month", "2025-12")
        dest_path = f"{self.config.storage.delta_dir}/scored_targets"

        self.delta_manager.write_idempotent_partition(
            df=data,
            path=dest_path,
            partition_col="snapshot_month",
            partition_val=snapshot_month,
        )
        version = self.delta_manager.get_latest_version(dest_path)

        return StageResult(
            stage_name=self.stage_name,
            run_id=kwargs.get("run_id", ""),
            snapshot_month=snapshot_month,
            status="SUCCESS",
            output_rows=data.count(),
            delta_versions={"scored_targets": version},
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    _score_worker(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        config_json=args.config_json,
    )
