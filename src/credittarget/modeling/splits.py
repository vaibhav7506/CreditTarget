"""Strict Out-of-Time (OOT) temporal train, validation, and holdout splitting."""

from __future__ import annotations

import pandas as pd

from credittarget.config import AppConfig
from credittarget.exceptions import DataQualityError


def validate_temporal_ordering(
    train_months: list[str],
    val_months: list[str],
    oot_months: list[str],
) -> None:
    """Asserts strict temporal order: max(train) < min(val) < min(oot)."""
    if not train_months or not val_months or not oot_months:
        raise DataQualityError(
            "Train, validation, and OOT split month lists must all be non-empty."
        )

    max_train = max(train_months)
    min_val = min(val_months)
    max_val = max(val_months)
    min_oot = min(oot_months)

    if not (max_train < min_val):
        raise DataQualityError(
            f"Temporal leak: max(train) '{max_train}' must be strictly before min(val) '{min_val}'."
        )
    if not (max_val < min_oot):
        raise DataQualityError(
            f"Temporal leak: max(val) '{max_val}' must be strictly before min(oot) '{min_oot}'."
        )


def get_temporal_splits(
    df: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a pandas DataFrame into Train (M1-8), Val/Calibration (M9), and OOT Holdout (M10-12)."""
    train_months = config.splits.train_months
    val_months = config.splits.val_months
    oot_months = config.splits.oot_months

    validate_temporal_ordering(train_months, val_months, oot_months)

    if "snapshot_month" not in df.columns:
        raise DataQualityError("DataFrame missing required 'snapshot_month' partitioning column.")

    train_df = df[df["snapshot_month"].isin(train_months)].copy()
    val_df = df[df["snapshot_month"].isin(val_months)].copy()
    oot_df = df[df["snapshot_month"].isin(oot_months)].copy()

    if train_df.empty or val_df.empty or oot_df.empty:
        raise DataQualityError(
            f"Temporal split produced an empty partition: "
            f"train={len(train_df)}, val={len(val_df)}, oot={len(oot_df)}"
        )

    return train_df, val_df, oot_df
