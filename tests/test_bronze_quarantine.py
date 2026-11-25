"""Tests for BronzeStage: schema enforcement, quarantine routing, and idempotency."""

from __future__ import annotations

from pyspark.sql import SparkSession

from credittarget.common.delta_io import DeltaTableManager
from credittarget.config import AppConfig
from credittarget.data_gen.synthetic_bank import generate_synthetic_bank_data
from credittarget.pipelines.bronze import BronzeStage


def test_bronze_stage_quarantine_routing(
    spark: SparkSession,
    app_config: AppConfig,
) -> None:
    """Verifies that synthetic data invalid records are split into quarantine Delta table."""
    raw_data = generate_synthetic_bank_data(spark, app_config)
    stage = BronzeStage(spark, app_config, raw_tables=raw_data)

    result = stage.run(run_id="test_bronze_run_1")
    assert result.status == "SUCCESS"
    assert result.output_rows > 0
    assert result.quarantine_rows > 0

    # Inspect bronze_quarantine Delta table
    delta_mgr = DeltaTableManager(spark)
    quarantine_path = f"{app_config.storage.delta_dir}/bronze_quarantine"
    q_df = delta_mgr.read_table(quarantine_path)
    assert q_df.count() == result.quarantine_rows

    reasons = [r["rejection_reason"] for r in q_df.select("rejection_reason").distinct().collect()]
    assert any("INVALID_UTILIZATION" in r or "NULL_CUSTOMER_ID" in r for r in reasons)


def test_bronze_stage_idempotent_rerun(
    spark: SparkSession,
    app_config: AppConfig,
) -> None:
    """Verifies that re-running BronzeStage produces identical output counts with zero duplicates."""
    raw_data = generate_synthetic_bank_data(spark, app_config)
    stage = BronzeStage(spark, app_config, raw_tables=raw_data)

    res1 = stage.run(run_id="run_idempotent_1")
    res2 = stage.run(run_id="run_idempotent_2")

    assert res1.status == "SUCCESS"
    assert res2.status == "SUCCESS"
    assert res1.output_rows == res2.output_rows

    delta_mgr = DeltaTableManager(spark)
    cust_df = delta_mgr.read_table(f"{app_config.storage.delta_dir}/bronze_customers")
    assert cust_df.count() == app_config.datagen.num_customers
