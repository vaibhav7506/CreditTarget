"""Comprehensive tests for common components: DeltaTableManager, DataQualityChecker, spark_utils, ArtifactStore, BasePipelineStage."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from credittarget.common.artifact_store import ArtifactStore, sha256_dict, sha256_text
from credittarget.common.base_stage import BasePipelineStage, StageResult
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.dq import DataQualityChecker
from credittarget.common.spark_utils import (
    assign_bins_by_edges,
    compute_reference_bin_edges,
    deduplicate_latest,
    export_physical_plan,
    safe_broadcast_join,
)
from credittarget.config import AppConfig
from credittarget.exceptions import DataQualityError, LeakageDetectedError
from credittarget.schemas.contracts import (
    CUSTOMERS_DIM_SCHEMA,
    compute_schema_hash,
)


def test_schema_hash_deterministic() -> None:
    """Verify compute_schema_hash produces deterministic SHA-256 digests."""
    hash1 = compute_schema_hash(CUSTOMERS_DIM_SCHEMA)
    hash2 = compute_schema_hash(CUSTOMERS_DIM_SCHEMA)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_artifact_store_local(artifact_store: ArtifactStore) -> None:
    """Verify local text and JSON atomic read/write and existence checks."""
    rel_text = "test_reports/memo.txt"
    artifact_store.write_text(rel_text, "Hello CreditTarget")
    assert artifact_store.exists(rel_text)
    assert artifact_store.read_text(rel_text) == "Hello CreditTarget"

    rel_json = "test_metrics/run_metrics.json"
    payload = {"gini": 0.52, "ks": 41.5, "model": "challenger"}
    artifact_store.write_json(rel_json, payload)
    read_back = artifact_store.read_json(rel_json)
    assert read_back == payload

    assert sha256_text("bank") == sha256_text("bank")
    assert sha256_dict({"a": 1, "b": 2}) == sha256_dict({"b": 2, "a": 1})


def test_delta_table_manager_lifecycle(
    spark: SparkSession,
    delta_manager: DeltaTableManager,
    tmp_workspace: Any,
) -> None:
    """Verify Delta write, overwrite, version history, partition replacement, and merge."""
    table_path = str(tmp_workspace / "delta_test" / "customers")

    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), False),
            StructField("region", StringType(), False),
            StructField("score", DoubleType(), False),
        ]
    )

    # 1. Overwrite write
    rows1 = [(1, "Alice", "NORTH", 0.8), (2, "Bob", "SOUTH", 0.5)]
    df1 = spark.createDataFrame(rows1, schema)
    delta_manager.write_overwrite_table(df1, table_path)

    read1 = delta_manager.read_table(table_path)
    assert read1.count() == 2
    assert delta_manager.get_latest_version(table_path) >= 0

    # 2. Idempotent partition replace
    part_table_path = str(tmp_workspace / "delta_test" / "partitioned_customers")
    part_rows = [(1, "Alice", "2025-01"), (2, "Bob", "2025-01"), (3, "Charlie", "2025-02")]
    part_schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), False),
            StructField("month", StringType(), False),
        ]
    )
    df_part = spark.createDataFrame(part_rows, part_schema)
    delta_manager.write_idempotent_partition(df_part, part_table_path, "month", "2025-01")

    # Replace partition 2025-01 with 1 row
    replacement_rows = [(10, "Alice_Updated", "2025-01")]
    df_replace = spark.createDataFrame(replacement_rows, part_schema)
    delta_manager.write_idempotent_partition(df_replace, part_table_path, "month", "2025-01")

    read_part = delta_manager.read_table(part_table_path)
    # Total count should be 2 (1 in 2025-01, 1 in 2025-02)
    assert read_part.filter(F.col("month") == "2025-01").count() == 1

    # 3. Upsert merge
    merge_rows = [(1, "Alice_V2", "NORTH", 0.95), (3, "Charlie", "EAST", 0.6)]
    df_merge = spark.createDataFrame(merge_rows, schema)
    delta_manager.upsert_merge(df_merge, table_path, merge_keys=["id"])

    read_merged = delta_manager.read_table(table_path)
    assert read_merged.count() == 3
    alice_row = read_merged.filter(F.col("id") == 1).collect()[0]
    assert alice_row["name"] == "Alice_V2"
    assert alice_row["score"] == 0.95


def test_delta_split_and_quarantine(
    spark: SparkSession,
    delta_manager: DeltaTableManager,
    tmp_workspace: Any,
) -> None:
    """Verify invalid rows are routed to quarantine Delta table with reason and metadata."""
    schema = StructType(
        [
            StructField("cust_id", StringType(), True),
            StructField("credit_limit", DoubleType(), False),
        ]
    )
    rows = [
        ("C1", 5000.0),  # Valid
        (None, 3000.0),  # Null key
        ("C3", -500.0),  # Negative credit limit
        ("C4", 10000.0),  # Valid
    ]
    df = spark.createDataFrame(rows, schema)
    quarantine_path = str(tmp_workspace / "delta_test" / "quarantine")

    invalid_expr_map = {
        "NULL_KEY": F.col("cust_id").isNull(),
        "NEGATIVE_LIMIT": F.col("credit_limit") < 0.0,
    }

    valid_df, quarantined_count = delta_manager.split_and_quarantine(
        df=df,
        invalid_expr_map=invalid_expr_map,
        quarantine_path=quarantine_path,
        batch_id="batch_001",
        source_table="customers_test",
    )

    assert valid_df.count() == 2
    assert quarantined_count == 2

    # Verify quarantine Delta table
    quarantined_df = delta_manager.read_table(quarantine_path)
    assert quarantined_df.count() == 2
    reasons = [row["rejection_reason"] for row in quarantined_df.collect()]
    assert "NULL_KEY" in reasons
    assert "NEGATIVE_LIMIT" in reasons


def test_data_quality_checker(spark: SparkSession) -> None:
    """Verify DataQualityChecker assertions for empty datasets, keys, nulls, bounds, and PIT."""
    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("val", DoubleType(), True),
            StructField("event_ts", TimestampType(), False),
            StructField("cutoff_ts", TimestampType(), False),
        ]
    )

    # 1. Non-empty
    empty_df = spark.createDataFrame([], schema)
    with pytest.raises(DataQualityError, match="is empty"):
        DataQualityChecker.assert_non_empty(empty_df, "empty_table")

    # 2. Unique keys
    dup_rows = [
        (1, 10.0, "2025-01-01 10:00:00", "2025-01-01 12:00:00"),
        (1, 20.0, "2025-01-01 10:00:00", "2025-01-01 12:00:00"),
    ]
    dup_df = spark.createDataFrame(dup_rows, ["id", "val", "event_ts", "cutoff_ts"])
    with pytest.raises(DataQualityError, match="duplicate rows detected"):
        DataQualityChecker.assert_unique_keys(dup_df, ["id"], "dup_table")

    # 3. Max null rate
    null_rows = [(1, None), (2, None), (3, 5.0), (4, 10.0)]
    null_df = spark.createDataFrame(null_rows, ["id", "val"])
    # Null rate is 2/4 = 0.50
    DataQualityChecker.assert_max_null_rate(null_df, {"val": 0.60}, "null_table")
    with pytest.raises(DataQualityError, match="exceeds threshold"):
        DataQualityChecker.assert_max_null_rate(null_df, {"val": 0.10}, "null_table")

    # 4. Numeric bounds
    bounds_rows = [(1, 10.0), (2, 50.0), (3, 120.0)]
    bounds_df = spark.createDataFrame(bounds_rows, ["id", "val"])
    DataQualityChecker.assert_numeric_bounds(bounds_df, {"val": (5.0, 150.0)}, "bounds_table")
    with pytest.raises(DataQualityError, match="exceeds upper bound"):
        DataQualityChecker.assert_numeric_bounds(bounds_df, {"val": (5.0, 100.0)}, "bounds_table")

    # 5. Point in time
    pit_rows = [
        ("2025-01-01 11:00:00", "2025-01-01 12:00:00"),  # Valid (before cutoff)
        ("2025-01-01 13:00:00", "2025-01-01 12:00:00"),  # Leakage (after cutoff)
    ]
    pit_df = (
        spark.createDataFrame(pit_rows, ["event_ts_str", "cutoff_ts_str"])
        .withColumn("event_ts", F.to_timestamp("event_ts_str"))
        .withColumn("cutoff_ts", F.to_timestamp("cutoff_ts_str"))
    )
    with pytest.raises(LeakageDetectedError, match="violated point-in-time invariant"):
        DataQualityChecker.assert_point_in_time(pit_df, "event_ts", "cutoff_ts", "pit_table")


def test_spark_utils(spark: SparkSession, artifact_store: ArtifactStore) -> None:
    """Verify deduplication, broadcast join, quantile binning, and plan export."""
    # 1. Deduplicate latest
    rows = [
        ("C1", "2025-01-01 10:00:00", 100.0),
        ("C1", "2025-01-01 12:00:00", 250.0),  # Latest
        ("C2", "2025-01-01 08:00:00", 50.0),
    ]
    df = spark.createDataFrame(rows, ["customer_id", "ts", "amount"])
    deduped = deduplicate_latest(df, ["customer_id"], "ts")
    assert deduped.count() == 2
    c1 = deduped.filter(F.col("customer_id") == "C1").collect()[0]
    assert c1["amount"] == 250.0

    # 2. Safe broadcast join
    dim_rows = [("BASIC", 0.18), ("PLUS", 0.14)]
    dim_df = spark.createDataFrame(dim_rows, ["segment", "yield"])
    fact_rows = [("C1", "BASIC"), ("C2", "PLUS"), ("C3", "PREMIER")]
    fact_df = spark.createDataFrame(fact_rows, ["cust_id", "segment"])
    joined = safe_broadcast_join(fact_df, dim_df, on="segment", how="left")
    assert joined.count() == 3

    # 3. Bin edges and assignment
    num_rows = [(float(x),) for x in range(1, 101)]
    df_nums = spark.createDataFrame(num_rows, ["score"])
    edges = compute_reference_bin_edges(df_nums, "score", num_bins=4)
    assert edges[0] == -float("inf")
    assert edges[-1] == float("inf")
    assert len(edges) >= 3

    binned_df = assign_bins_by_edges(df_nums, "score", edges, "bin_idx")
    assert "bin_idx" in binned_df.columns
    assert binned_df.filter(F.col("bin_idx") == -1).count() == 0

    # 4. Plan export
    plan = export_physical_plan(joined, artifact_store, "test_plans/test_plan.txt")
    assert isinstance(plan, str)
    assert artifact_store.exists("test_plans/test_plan.txt")


def test_base_pipeline_stage_runner(app_config: AppConfig) -> None:
    """Verify BasePipelineStage executes lifecycle methods with error handling and timing."""

    class MockStage(BasePipelineStage):
        def extract(self, *args: Any, **kwargs: Any) -> list[int]:
            return [1, 2, 3]

        def transform(self, data: Any, *args: Any, **kwargs: Any) -> list[int]:
            return [x * 2 for x in data]

        def validate(self, data: Any, *args: Any, **kwargs: Any) -> None:
            assert len(data) == 3

        def load(self, data: Any, *args: Any, **kwargs: Any) -> StageResult:
            return StageResult(
                stage_name=self.stage_name,
                run_id="",
                output_rows=len(data),
                status="SUCCESS",
            )

    stage = MockStage(app_config)
    res = stage.run(snapshot_month="2025-01")
    assert res.status == "SUCCESS"
    assert res.output_rows == 3
    assert res.snapshot_month == "2025-01"
    assert res.duration_sec >= 0.0
    assert len(res.run_id) > 0
