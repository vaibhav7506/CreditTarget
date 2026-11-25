"""Delta Lake I/O manager with atomic replaceWhere, merge upserts, and quarantine routing."""

from __future__ import annotations

import pyspark.sql.functions as F
from delta.tables import DeltaTable
from pyspark.sql import Column, DataFrame, SparkSession

from credittarget.logging_utils import get_logger

logger = get_logger("DeltaTableManager")


class DeltaTableManager:
    """Manages Delta table reading, writing, version tracking, and quarantine routing."""

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    def read_table(self, path: str, version: int | None = None) -> DataFrame:
        """Reads a Delta table optionally pinned to a historical version."""
        reader = self.spark.read.format("delta")
        if version is not None:
            reader = reader.option("versionAsOf", version)
        return reader.load(path)

    def write_overwrite_table(
        self,
        df: DataFrame,
        path: str,
        partition_cols: list[str] | None = None,
    ) -> None:
        """Atomically overwrites the entire Delta table."""
        writer = df.write.format("delta").mode("overwrite")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.save(path)

    def write_idempotent_partition(
        self,
        df: DataFrame,
        path: str,
        partition_col: str,
        partition_val: str,
    ) -> None:
        """Idempotently replaces only the specified partition using Delta replaceWhere."""
        is_delta = False
        try:
            is_delta = DeltaTable.isDeltaTable(self.spark, path)
        except Exception:
            is_delta = False

        if is_delta:
            condition = f"{partition_col} = '{partition_val}'"
            (
                df.write.format("delta")
                .mode("overwrite")
                .option("replaceWhere", condition)
                .save(path)
            )
        else:
            (df.write.format("delta").mode("overwrite").partitionBy(partition_col).save(path))

    def upsert_merge(
        self,
        df: DataFrame,
        path: str,
        merge_keys: list[str],
    ) -> None:
        """Merges DataFrame into Delta table based on composite merge keys."""
        is_delta = False
        try:
            is_delta = DeltaTable.isDeltaTable(self.spark, path)
        except Exception:
            is_delta = False

        if not is_delta:
            self.write_overwrite_table(df, path)
            return

        delta_table = DeltaTable.forPath(self.spark, path)
        join_cond = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
        (
            delta_table.alias("target")
            .merge(df.alias("source"), join_cond)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def split_and_quarantine(
        self,
        df: DataFrame,
        invalid_expr_map: dict[str, Column],
        quarantine_path: str,
        batch_id: str,
        source_table: str,
    ) -> tuple[DataFrame, int]:
        """Evaluates invalid conditions, routes matching rows to quarantine, and returns valid rows."""
        if not invalid_expr_map:
            return df, 0

        # Build chained rejection reason
        reason_col: Column | None = None
        for reason, cond in invalid_expr_map.items():
            if reason_col is None:
                reason_col = F.when(cond, F.lit(reason))
            else:
                reason_col = reason_col.when(cond, F.lit(reason))
        assert reason_col is not None
        reason_col = reason_col.otherwise(F.lit(None))

        annotated_df = df.withColumn("_rejection_reason", reason_col)
        valid_df = annotated_df.filter(F.col("_rejection_reason").isNull()).drop(
            "_rejection_reason"
        )
        invalid_df = annotated_df.filter(F.col("_rejection_reason").isNotNull())

        invalid_count = invalid_df.count()
        if invalid_count > 0:
            logger.warn(
                "quarantine_rows_detected",
                source_table=source_table,
                quarantined_count=invalid_count,
                batch_id=batch_id,
            )
            # Create quarantined payload record
            quarantine_records = invalid_df.select(
                F.to_json(F.struct("*")).alias("raw_payload"),
                F.col("_rejection_reason").alias("rejection_reason"),
                F.lit(source_table).alias("source_table"),
                F.lit(batch_id).alias("source_batch_id"),
                F.current_timestamp().alias("quarantined_at_utc"),
            )
            (quarantine_records.write.format("delta").mode("append").save(quarantine_path))

        return valid_df, invalid_count

    def get_latest_version(self, path: str) -> int:
        """Returns the latest Delta version or 0 if table does not exist."""
        try:
            if DeltaTable.isDeltaTable(self.spark, path):
                dt = DeltaTable.forPath(self.spark, path)
                history_df = dt.history(1).select("version").collect()
                if history_df:
                    return int(history_df[0]["version"])
        except Exception:
            return 0
        return 0
