"""Fairness and demographic disparate impact analysis across protected/sensitive customer attributes."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_group_targeting_rates(
    df: pd.DataFrame,
    group_col: str,
    decision_col: str = "target_decision",
    target_val: str = "TARGET",
) -> dict[str, Any]:
    """Computes total counts, targeting counts, and targeting rates per group."""
    if group_col not in df.columns or decision_col not in df.columns:
        return {"groups": {}, "disparate_impact_ratio": 1.0}

    grouped = df.groupby(group_col)
    groups_info: dict[str, dict[str, Any]] = {}
    rates: list[float] = []

    for name, group in grouped:
        total = len(group)
        targeted = int((group[decision_col] == target_val).sum())
        rate = round(targeted / total, 4) if total > 0 else 0.0
        rates.append(rate)
        groups_info[str(name)] = {
            "total_customers": total,
            "targeted_customers": targeted,
            "targeting_rate": rate,
        }

    # Disparate impact ratio: min_rate / max_rate
    min_rate = min(rates) if rates else 0.0
    max_rate = max(rates) if rates else 1.0
    di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

    return {
        "groups": groups_info,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "disparate_impact_ratio": di_ratio,
    }


def compute_fairness_disparate_impact(
    df: pd.DataFrame,
    protected_cols: list[str] | None = None,
    decision_col: str = "target_decision",
) -> dict[str, Any]:
    """Evaluates fairness and disparate impact across region_code and age_band."""
    cols = protected_cols or ["region_code", "age_band"]
    fairness_report: dict[str, Any] = {}
    min_overall_di = 1.0

    for c in cols:
        if c in df.columns:
            res = compute_group_targeting_rates(df, c, decision_col=decision_col)
            fairness_report[c] = res
            if res["disparate_impact_ratio"] < min_overall_di:
                min_overall_di = res["disparate_impact_ratio"]

    fairness_report["overall_min_disparate_impact"] = min_overall_di
    return fairness_report
