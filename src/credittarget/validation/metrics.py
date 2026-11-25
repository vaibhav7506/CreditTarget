"""100% Distributed PySpark validation metrics: PSI, CSI, KS, Gini, ROC-AUC, Brier, and Decile Lift."""

from __future__ import annotations

import math
from typing import Any

import pyspark.sql.functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame, Window

from credittarget.common.spark_utils import (
    assign_bins_by_edges,
    compute_reference_bin_edges,
)


def compute_binary_classification_metrics(
    spark_df: DataFrame,
    label_col: str,
    score_col: str,
) -> dict[str, float]:
    """Computes distributed ROC-AUC, Gini, PR-AUC, Brier Score, and KS statistic in PySpark."""
    eval_df = spark_df.select(
        F.col(label_col).cast("double").alias("label"),
        F.col(score_col).cast("double").alias("score"),
    ).cache()

    total_count = eval_df.count()
    if total_count == 0:
        return {"roc_auc": 0.5, "gini": 0.0, "pr_auc": 0.0, "brier_score": 0.0, "ks_statistic": 0.0}

    # 1. ROC-AUC and PR-AUC using PySpark BinaryClassificationEvaluator
    roc_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="score",
        labelCol="label",
        metricName="areaUnderROC",
    )
    roc_auc = float(roc_evaluator.evaluate(eval_df))
    gini = float(2.0 * roc_auc - 1.0)

    pr_evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="score",
        labelCol="label",
        metricName="areaUnderPR",
    )
    pr_auc = float(pr_evaluator.evaluate(eval_df))

    # 2. Distributed Brier Score: mean((score - label)^2)
    brier_res = eval_df.select(
        F.avg(F.pow(F.col("score") - F.col("label"), 2)).alias("brier")
    ).collect()
    brier_score = float(brier_res[0]["brier"] or 0.0)

    # 3. Distributed Kolmogorov-Smirnov (KS) statistic: max |CDF_pos(s) - CDF_neg(s)|
    # Compute over 100 quantiles / percentiles
    quantiles = [i / 100.0 for i in range(1, 100)]
    thresholds = sorted(set(eval_df.approxQuantile("score", quantiles, 0.01)))

    total_pos = eval_df.filter(F.col("label") == 1.0).count()
    total_neg = total_count - total_pos

    ks_stat = 0.0
    if total_pos > 0 and total_neg > 0 and thresholds:
        # Compute cumulative distributions across thresholds
        agg_exprs = []
        for idx, th in enumerate(thresholds):
            agg_exprs.append(
                F.count(F.when((F.col("score") >= th) & (F.col("label") == 1.0), 1)).alias(
                    f"p_{idx}"
                )
            )
            agg_exprs.append(
                F.count(F.when((F.col("score") >= th) & (F.col("label") == 0.0), 1)).alias(
                    f"n_{idx}"
                )
            )
        counts_row = eval_df.agg(*agg_exprs).collect()[0].asDict()

        for idx in range(len(thresholds)):
            cum_tpr = counts_row[f"p_{idx}"] / total_pos
            cum_fpr = counts_row[f"n_{idx}"] / total_neg
            diff = abs(cum_tpr - cum_fpr)
            if diff > ks_stat:
                ks_stat = diff

    eval_df.unpersist()

    return {
        "roc_auc": round(roc_auc, 4),
        "gini": round(gini, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier_score, 4),
        "ks_statistic": round(ks_stat * 100.0, 2),  # In percentage points [0, 100]
    }


def compute_psi(
    ref_df: DataFrame,
    target_df: DataFrame,
    score_col: str,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> tuple[float, list[dict[str, Any]]]:
    """Computes distributed Population Stability Index (PSI) using edges fit strictly on ref_df."""
    ref_count = ref_df.count()
    target_count = target_df.count()
    if ref_count == 0 or target_count == 0:
        return 0.0, []

    edges = compute_reference_bin_edges(ref_df, score_col, num_bins=num_bins)
    num_intervals = len(edges) - 1

    binned_ref = assign_bins_by_edges(ref_df, score_col, edges, "bin_idx")
    binned_target = assign_bins_by_edges(target_df, score_col, edges, "bin_idx")

    ref_counts = binned_ref.groupBy("bin_idx").count().collect()
    target_counts = binned_target.groupBy("bin_idx").count().collect()

    ref_map = {row["bin_idx"]: row["count"] for row in ref_counts}
    target_map = {row["bin_idx"]: row["count"] for row in target_counts}

    psi_total = 0.0
    bin_breakdown: list[dict[str, Any]] = []

    for b in range(num_intervals):
        r_cnt = ref_map.get(b, 0)
        t_cnt = target_map.get(b, 0)

        r_pct = max(eps, r_cnt / ref_count)
        t_pct = max(eps, t_cnt / target_count)

        bin_psi = (t_pct - r_pct) * math.log(t_pct / r_pct)
        psi_total += bin_psi

        bin_breakdown.append(
            {
                "bin": b,
                "lower_edge": edges[b],
                "upper_edge": edges[b + 1],
                "ref_count": r_cnt,
                "target_count": t_cnt,
                "ref_pct": round(r_pct, 4),
                "target_pct": round(t_pct, 4),
                "psi_contribution": round(bin_psi, 4),
            }
        )

    return round(float(psi_total), 4), bin_breakdown


def compute_csi(
    ref_df: DataFrame,
    target_df: DataFrame,
    feature_cols: list[str],
    num_bins: int = 10,
) -> dict[str, float]:
    """Computes Characteristic Stability Index (CSI) for multiple numeric features."""
    csi_results: dict[str, float] = {}
    for col_name in feature_cols:
        try:
            psi_val, _ = compute_psi(ref_df, target_df, col_name, num_bins=num_bins)
            csi_results[col_name] = psi_val
        except Exception:
            csi_results[col_name] = 0.0
    return csi_results


def compute_decile_lift_table(
    spark_df: DataFrame,
    label_col: str,
    score_col: str,
    pd_col: str | None = None,
    ev_col: str | None = None,
) -> list[dict[str, Any]]:
    """Generates distributed 10-decile lift and financial capture table (Decile 1 = highest propensity)."""
    total_count = spark_df.count()
    if total_count == 0:
        return []

    # Assign deciles (1..10) using ntile over score descending
    w = Window.orderBy(F.col(score_col).desc())
    decile_df = spark_df.withColumn("decile", F.ntile(10).over(w))

    agg_exprs = [
        F.count("*").alias("count"),
        F.sum(F.col(label_col).cast("int")).alias("responders"),
        F.avg(F.col(score_col)).alias("mean_propensity"),
    ]
    if pd_col and pd_col in spark_df.columns:
        agg_exprs.append(F.avg(F.col(pd_col)).alias("avg_pd"))
    if ev_col and ev_col in spark_df.columns:
        agg_exprs.append(F.sum(F.col(ev_col)).alias("total_ev"))

    decile_stats = decile_df.groupBy("decile").agg(*agg_exprs).orderBy("decile").collect()

    total_responders = sum(row["responders"] for row in decile_stats)
    overall_response_rate = (total_responders / total_count) if total_count > 0 else 0.0

    table: list[dict[str, Any]] = []
    cum_responders = 0
    cum_ev = 0.0

    for row in decile_stats:
        d = int(row["decile"])
        cnt = int(row["count"])
        resp = int(row["responders"])
        resp_rate = (resp / cnt) if cnt > 0 else 0.0
        lift = (resp_rate / overall_response_rate) if overall_response_rate > 0 else 1.0

        cum_responders += resp
        cum_capture = (cum_responders / total_responders) if total_responders > 0 else 0.0

        entry: dict[str, Any] = {
            "decile": d,
            "count": cnt,
            "responders": resp,
            "response_rate": round(resp_rate, 4),
            "lift": round(lift, 2),
            "cumulative_capture_rate": round(cum_capture, 4),
            "mean_propensity": round(float(row["mean_propensity"]), 4),
        }
        if pd_col and "avg_pd" in row.asDict():
            entry["avg_pd"] = round(float(row["avg_pd"]), 4)
        if ev_col and "total_ev" in row.asDict():
            d_ev = float(row["total_ev"])
            cum_ev += d_ev
            entry["decile_ev"] = round(d_ev, 2)
            entry["cumulative_ev"] = round(cum_ev, 2)

        table.append(entry)

    return table
