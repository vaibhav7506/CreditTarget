"""Reusable distributed Spark Data Quality assertions for CreditTarget."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from credittarget.exceptions import DataQualityError, LeakageDetectedError


class DataQualityChecker:
    """Performs strict runtime data quality assertions on PySpark DataFrames."""

    @staticmethod
    def assert_non_empty(df: DataFrame, name: str) -> int:
        """Asserts dataframe is non-empty and returns the row count."""
        count = df.count()
        if count == 0:
            raise DataQualityError(f"Dataset '{name}' is empty (0 rows).")
        return count

    @staticmethod
    def assert_unique_keys(df: DataFrame, key_cols: list[str], name: str) -> None:
        """Asserts composite primary keys are strictly unique."""
        total_count = df.count()
        distinct_count = df.select(*key_cols).distinct().count()
        if total_count != distinct_count:
            diff = total_count - distinct_count
            raise DataQualityError(
                f"Dataset '{name}' failed uniqueness check on keys {key_cols}: "
                f"{diff} duplicate rows detected ({total_count} total vs {distinct_count} distinct)."
            )

    @staticmethod
    def assert_max_null_rate(
        df: DataFrame,
        max_rates: dict[str, float],
        name: str,
        total_count: int | None = None,
    ) -> None:
        """Asserts null rate for specified columns does not exceed max allowed threshold."""
        if not max_rates:
            return
        n = total_count if total_count is not None else df.count()
        if n == 0:
            return

        agg_exprs = [
            F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1)).alias(c)
            for c in max_rates
            if c in df.columns
        ]
        if not agg_exprs:
            return

        null_counts = df.agg(*agg_exprs).collect()[0].asDict()
        violations: list[str] = []
        for col_name, max_rate in max_rates.items():
            if col_name in null_counts:
                cnt = null_counts[col_name]
                rate = cnt / n
                if rate > max_rate:
                    violations.append(
                        f"Column '{col_name}' null rate {rate:.4f} exceeds threshold {max_rate:.4f} ({cnt}/{n})"
                    )

        if violations:
            raise DataQualityError(
                f"Dataset '{name}' violated null rate constraints:\n" + "\n".join(violations)
            )

    @staticmethod
    def assert_numeric_bounds(
        df: DataFrame,
        bounds: dict[str, tuple[float | None, float | None]],
        name: str,
    ) -> None:
        """Asserts numeric columns stay within specified [min, max] inclusive bounds."""
        violations: list[str] = []
        agg_exprs = []
        bound_cols = [c for c in bounds if c in df.columns]
        for c in bound_cols:
            agg_exprs.extend(
                [
                    F.min(F.col(c)).alias(f"{c}__min"),
                    F.max(F.col(c)).alias(f"{c}__max"),
                ]
            )
        if not agg_exprs:
            return

        stats: dict[str, Any] = df.agg(*agg_exprs).collect()[0].asDict()
        for col_name in bound_cols:
            lower, upper = bounds[col_name]
            actual_min = stats.get(f"{col_name}__min")
            actual_max = stats.get(f"{col_name}__max")
            if actual_min is not None and lower is not None and actual_min < lower:
                violations.append(
                    f"Column '{col_name}' min value {actual_min} is below lower bound {lower}"
                )
            if actual_max is not None and upper is not None and actual_max > upper:
                violations.append(
                    f"Column '{col_name}' max value {actual_max} exceeds upper bound {upper}"
                )

        if violations:
            raise DataQualityError(
                f"Dataset '{name}' violated numeric bounds constraints:\n" + "\n".join(violations)
            )

    @staticmethod
    def assert_point_in_time(
        df: DataFrame,
        event_ts_col: str,
        cutoff_ts_col: str,
        name: str,
    ) -> None:
        """Asserts point-in-time invariant: zero events occur after the cutoff snapshot timestamp."""
        leaked_count = df.filter(F.col(event_ts_col) > F.col(cutoff_ts_col)).count()
        if leaked_count > 0:
            raise LeakageDetectedError(
                f"Dataset '{name}' violated point-in-time invariant: {leaked_count} events "
                f"have {event_ts_col} > {cutoff_ts_col}."
            )
