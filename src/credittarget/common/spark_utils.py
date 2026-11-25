"""Shared PySpark utilities for deduplication, broadcast joins, quantile binning, and plan export."""

from __future__ import annotations

import math

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame, Window

from credittarget.common.artifact_store import ArtifactStore
from credittarget.exceptions import DataQualityError


def deduplicate_latest(
    df: DataFrame,
    key_cols: list[str],
    order_col: str,
) -> DataFrame:
    """Deduplicates a DataFrame by composite key, preserving the most recent record."""
    window_spec = Window.partitionBy(*key_cols).orderBy(F.col(order_col).desc())
    return (
        df.withColumn("_rn", F.row_number().over(window_spec)).filter(F.col("_rn") == 1).drop("_rn")
    )


def safe_broadcast_join(
    fact_df: DataFrame,
    dim_df: DataFrame,
    on: str | list[str],
    how: str = "left",
) -> DataFrame:
    """Executes a join with an explicit broadcast hint on the dimension table."""
    return fact_df.join(F.broadcast(dim_df), on=on, how=how)


def compute_reference_bin_edges(
    df: DataFrame,
    col_name: str,
    num_bins: int = 10,
) -> list[float]:
    """Computes reference bin edges via approxQuantile with deduplicated interior edges and [-inf, +inf]."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    probabilities = [i / num_bins for i in range(1, num_bins)]
    raw_quantiles = df.approxQuantile(col_name, probabilities, 0.01)

    # Filter out NaNs and sort unique interior edges
    valid_quantiles = sorted({float(q) for q in raw_quantiles if not math.isnan(q)})

    edges = [-float("inf")] + valid_quantiles + [float("inf")]
    return edges


def assign_bins_by_edges(
    df: DataFrame,
    col_name: str,
    edges: list[float],
    output_col: str = "bin_index",
) -> DataFrame:
    """Assigns integer bin indices [0, len(edges)-2] according to precomputed reference edges."""
    if len(edges) < 2:
        raise ValueError("edges must contain at least 2 elements")

    # Chain when conditions
    bin_expr: Column | None = None
    num_intervals = len(edges) - 1

    for i in range(num_intervals):
        lower = edges[i]
        upper = edges[i + 1]
        if i == num_intervals - 1:
            # Last bin is inclusive of upper bound
            cond = (F.col(col_name) >= lower) & (F.col(col_name) <= upper)
        else:
            cond = (F.col(col_name) >= lower) & (F.col(col_name) < upper)

        if bin_expr is None:
            bin_expr = F.when(cond, F.lit(i))
        else:
            bin_expr = bin_expr.when(cond, F.lit(i))

    assert bin_expr is not None
    bin_expr = bin_expr.otherwise(F.lit(-1))
    return df.withColumn(output_col, bin_expr)


def export_physical_plan(
    df: DataFrame,
    artifact_store: ArtifactStore,
    rel_path: str,
    required_operators: list[str] | None = None,
) -> str:
    """Extracts physical execution plan, verifies required operators, and persists plan artifact."""
    # Obtain executed plan string from Spark query execution
    try:
        plan_str = str(df._jdf.queryExecution().executedPlan().toString())
    except Exception:
        plan_str = str(df._jdf.queryExecution().toString())

    if required_operators:
        missing_ops = [op for op in required_operators if op not in plan_str]
        if missing_ops:
            raise DataQualityError(
                f"Spark execution plan failed operator requirement check. Missing: {missing_ops}\n"
                f"Plan snippet:\n{plan_str[:500]}"
            )

    artifact_store.write_text(rel_path, plan_str)
    return plan_str
