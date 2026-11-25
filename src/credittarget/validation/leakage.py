"""Leakage detection: forbidden column scanner and univariate ROC-AUC anomaly detector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credittarget.exceptions import LeakageDetectedError
from credittarget.modeling.features import FORBIDDEN_LEAKAGE_COLUMNS


def assert_zero_leakage_columns(
    feature_cols: list[str],
    forbidden_cols: list[str] | None = None,
) -> None:
    """Verifies no target or leakage columns exist in the feature set."""
    forbidden = set(forbidden_cols or FORBIDDEN_LEAKAGE_COLUMNS)
    violations = [c for c in feature_cols if c in forbidden]
    if violations:
        raise LeakageDetectedError(
            f"Target leakage detected: forbidden columns in feature set: {violations}"
        )


def detect_univariate_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    max_auc_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Scans individual features for suspiciously high univariate predictive power (> 0.95 AUC)."""
    assert_zero_leakage_columns(feature_cols)

    y = np.ascontiguousarray(df[target_col].values, dtype=np.int32)
    if len(np.unique(y)) < 2:
        return []

    flagged: list[dict[str, Any]] = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        # Only evaluate numeric features for univariate AUC
        if not np.issubdtype(series.dtype, np.number):
            continue

        clean_series = series.fillna(series.median() if not series.empty else 0.0).values
        try:
            auc = roc_auc_score(y, clean_series)
            auc_effective = max(auc, 1.0 - auc)
            if auc_effective > max_auc_threshold:
                flagged.append(
                    {
                        "feature": col,
                        "univariate_auc": round(float(auc_effective), 4),
                        "status": "SUSPECTED_LEAKAGE",
                    }
                )
        except Exception:
            continue

    return flagged
