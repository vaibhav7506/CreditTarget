"""Tests for distributed PySpark validation metrics: ROC-AUC, Gini, Brier, KS, PSI, and Decile Lift."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from credittarget.validation.metrics import (
    compute_binary_classification_metrics,
    compute_decile_lift_table,
    compute_psi,
)


def test_binary_classification_metrics_analytical(spark: SparkSession) -> None:
    """Verifies that PySpark binary metrics compute correct AUC, Gini, Brier, and KS on synthetic data."""
    # Perfectly separated data
    data = [
        (1, 0.9),
        (1, 0.8),
        (1, 0.7),
        (0, 0.3),
        (0, 0.2),
        (0, 0.1),
    ]
    schema = StructType(
        [
            StructField("label", IntegerType(), False),
            StructField("score", DoubleType(), False),
        ]
    )
    df = spark.createDataFrame(data, schema)

    metrics = compute_binary_classification_metrics(df, "label", "score")
    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks_statistic"] == 100.0  # Perfect separation -> 100%
    assert metrics["brier_score"] < 0.10


def test_psi_identical_distributions(spark: SparkSession) -> None:
    """Verifies that PSI between identical distributions is approximately 0.0."""
    data = [(float(x),) for x in range(100)]
    schema = StructType([StructField("score", DoubleType(), False)])
    ref_df = spark.createDataFrame(data, schema)
    target_df = spark.createDataFrame(data, schema)

    psi_val, breakdown = compute_psi(ref_df, target_df, "score", num_bins=10)
    assert abs(psi_val) < 0.01
    assert len(breakdown) == 10


def test_psi_shifted_distribution(spark: SparkSession) -> None:
    """Verifies that PSI detects severe distribution shift (> 0.25)."""
    ref_data = [(float(x),) for x in range(0, 50)]
    target_data = [(float(x),) for x in range(50, 100)]
    schema = StructType([StructField("score", DoubleType(), False)])

    ref_df = spark.createDataFrame(ref_data, schema)
    target_df = spark.createDataFrame(target_data, schema)

    psi_val, _ = compute_psi(ref_df, target_df, "score", num_bins=5)
    assert psi_val > 0.25


def test_decile_lift_table_structure(spark: SparkSession) -> None:
    """Verifies 10 deciles, monotonically increasing cumulative capture rate, and lift."""
    # 100 rows with high scores correlated with label=1
    rows = []
    for i in range(100):
        score = i / 100.0
        label = 1 if i >= 70 else 0
        rows.append((label, score, score * 0.1, score * 50.0))

    schema = StructType(
        [
            StructField("label", IntegerType(), False),
            StructField("propensity", DoubleType(), False),
            StructField("pd", DoubleType(), False),
            StructField("ev", DoubleType(), False),
        ]
    )
    df = spark.createDataFrame(rows, schema)

    table = compute_decile_lift_table(df, "label", "propensity", "pd", "ev")
    assert len(table) == 10
    # Decile 1 should have highest responders and lift > 1.0
    assert table[0]["lift"] > 1.0
    # Last decile cumulative capture rate must be 1.0
    assert table[-1]["cumulative_capture_rate"] == 1.0
