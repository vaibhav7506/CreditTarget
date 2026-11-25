"""Silver Lakehouse stage: normalized deduped entities, NLP enrichment, and entity-level data quality."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from credittarget.common.base_stage import BasePipelineStage, StageResult
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.dq import DataQualityChecker
from credittarget.common.spark_utils import deduplicate_latest
from credittarget.config import AppConfig
from credittarget.pipelines.nlp_udf import get_nlp_scorer_udf


class SilverStage(BasePipelineStage):
    """Silver Stage normalizes, deduplicates entities, and enriches support notes with vectorized NLP."""

    def __init__(self, spark: SparkSession, config: AppConfig) -> None:
        super().__init__(config=config, stage_name="SilverStage")
        self.spark = spark
        self.delta_manager = DeltaTableManager(spark)

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, DataFrame]:
        """Extracts cleaned datasets from Bronze Delta tables."""
        delta_root = self.config.storage.delta_dir
        return {
            "bronze_customers": self.delta_manager.read_table(f"{delta_root}/bronze_customers"),
            "bronze_product_catalog": self.delta_manager.read_table(
                f"{delta_root}/bronze_product_catalog"
            ),
            "bronze_transactions": self.delta_manager.read_table(
                f"{delta_root}/bronze_transactions"
            ),
            "bronze_support_notes": self.delta_manager.read_table(
                f"{delta_root}/bronze_support_notes"
            ),
            "bronze_campaign_labels": self.delta_manager.read_table(
                f"{delta_root}/bronze_campaign_labels"
            ),
        }

    def transform(
        self, data: dict[str, DataFrame], *args: Any, **kwargs: Any
    ) -> dict[str, DataFrame]:
        """Deduplicates records, standardizes timestamps, and applies vectorized NLP UDF."""
        # 1. Silver Customers: deduplicated by customer_id
        silver_customers = deduplicate_latest(
            data["bronze_customers"],
            key_cols=["customer_id"],
            order_col="ingested_at_utc",
        )

        # 2. Silver Product Catalog: deduplicated by product_segment
        silver_product_catalog = deduplicate_latest(
            data["bronze_product_catalog"],
            key_cols=["product_segment"],
            order_col="ingested_at_utc",
        )

        # 3. Silver Transactions: deduplicated by txn_id
        silver_transactions = deduplicate_latest(
            data["bronze_transactions"],
            key_cols=["txn_id"],
            order_col="ingested_at_utc",
        ).withColumn("txn_ts", F.to_utc_timestamp(F.col("txn_ts"), "UTC"))

        # 4. Silver Support Notes: deduplicated by note_id and enriched via vectorized pandas_udf
        deduped_notes = deduplicate_latest(
            data["bronze_support_notes"],
            key_cols=["note_id"],
            order_col="ingested_at_utc",
        ).withColumn("note_ts", F.to_utc_timestamp(F.col("note_ts"), "UTC"))

        # Vectorized NLP enrichment
        scored_notes = deduped_notes.withColumn("nlp", get_nlp_scorer_udf(F.col("note_text")))
        silver_support_notes_nlp = (
            scored_notes.withColumn("sentiment_score", F.col("nlp.sentiment_score"))
            .withColumn("distress_flag", F.col("nlp.distress_flag"))
            .withColumn("complaint_severity", F.col("nlp.complaint_severity"))
            .withColumn("credit_intent_flag", F.col("nlp.credit_intent_flag"))
            .drop("nlp")
        )

        # 5. Silver Campaign Labels: deduplicated by (customer_id, snapshot_month)
        silver_campaign_labels = deduplicate_latest(
            data["bronze_campaign_labels"],
            key_cols=["customer_id", "snapshot_month"],
            order_col="ingested_at_utc",
        ).withColumn("snapshot_date", F.to_utc_timestamp(F.col("snapshot_date"), "UTC"))

        return {
            "silver_customers": silver_customers,
            "silver_product_catalog": silver_product_catalog,
            "silver_transactions": silver_transactions,
            "silver_support_notes_nlp": silver_support_notes_nlp,
            "silver_campaign_labels": silver_campaign_labels,
        }

    def validate(self, data: dict[str, DataFrame], *args: Any, **kwargs: Any) -> None:
        """Runs strict DataQualityChecker assertions across all Silver tables."""
        # Non-empty checks
        for name, df in data.items():
            DataQualityChecker.assert_non_empty(df, name)

        # Primary key uniqueness checks
        DataQualityChecker.assert_unique_keys(
            data["silver_customers"], ["customer_id"], "silver_customers"
        )
        DataQualityChecker.assert_unique_keys(
            data["silver_product_catalog"], ["product_segment"], "silver_product_catalog"
        )
        DataQualityChecker.assert_unique_keys(
            data["silver_transactions"], ["txn_id"], "silver_transactions"
        )
        DataQualityChecker.assert_unique_keys(
            data["silver_support_notes_nlp"], ["note_id"], "silver_support_notes_nlp"
        )
        DataQualityChecker.assert_unique_keys(
            data["silver_campaign_labels"],
            ["customer_id", "snapshot_month"],
            "silver_campaign_labels",
        )

        # Numeric bounds checks
        DataQualityChecker.assert_numeric_bounds(
            data["silver_customers"],
            {"credit_limit": (0.0, None)},
            "silver_customers",
        )
        DataQualityChecker.assert_numeric_bounds(
            data["silver_transactions"],
            {"utilization_ratio": (0.0, 1.5), "amount": (0.0, None)},
            "silver_transactions",
        )
        DataQualityChecker.assert_numeric_bounds(
            data["silver_support_notes_nlp"],
            {
                "sentiment_score": (-1.0, 1.0),
                "distress_flag": (0.0, 1.0),
                "complaint_severity": (0.0, 1.0),
                "credit_intent_flag": (0.0, 1.0),
            },
            "silver_support_notes_nlp",
        )

    def load(self, data: dict[str, DataFrame], *args: Any, **kwargs: Any) -> StageResult:
        """Persists validated Silver tables to Delta."""
        delta_root = self.config.storage.delta_dir
        delta_versions: dict[str, int] = {}
        total_rows = 0

        for tbl_name, df in data.items():
            path = f"{delta_root}/{tbl_name}"
            if tbl_name == "silver_campaign_labels":
                self.delta_manager.write_overwrite_table(
                    df, path, partition_cols=["snapshot_month"]
                )
            else:
                self.delta_manager.write_overwrite_table(df, path)
            delta_versions[tbl_name] = self.delta_manager.get_latest_version(path)
            total_rows += df.count()

        return StageResult(
            stage_name=self.stage_name,
            run_id=kwargs.get("run_id", ""),
            status="SUCCESS",
            output_rows=total_rows,
            delta_versions=delta_versions,
        )
