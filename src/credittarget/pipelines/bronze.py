"""Bronze Lakehouse stage: schema enforcement, quarantine routing, and raw ingestion."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from credittarget.common.base_stage import BasePipelineStage, StageResult
from credittarget.common.delta_io import DeltaTableManager
from credittarget.config import AppConfig
from credittarget.data_gen.synthetic_bank import generate_synthetic_bank_data


class BronzeStage(BasePipelineStage):
    """Bronze Stage ingests raw tables, applies quarantine rules, and stores to Bronze Delta tables."""

    def __init__(
        self,
        spark: SparkSession,
        config: AppConfig,
        raw_tables: dict[str, DataFrame] | None = None,
    ) -> None:
        super().__init__(config=config, stage_name="BronzeStage")
        self.spark = spark
        self.delta_manager = DeltaTableManager(spark)
        self.raw_tables = raw_tables

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, DataFrame]:
        """Extracts or receives the 5 raw datasets."""
        if self.raw_tables:
            return self.raw_tables
        return generate_synthetic_bank_data(self.spark, self.config)

    def transform(self, data: dict[str, DataFrame], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Applies validation expressions and splits out corrupt rows to quarantine."""
        run_id = kwargs.get("run_id", "batch_bronze")
        delta_root = self.config.storage.delta_dir
        quarantine_path = f"{delta_root}/bronze_quarantine"

        total_quarantined = 0
        clean_tables: dict[str, DataFrame] = {}

        # 1. Customers quarantine
        cust_invalid = {
            "NULL_CUSTOMER_ID": F.col("customer_id").isNull(),
            "NEGATIVE_CREDIT_LIMIT": F.col("credit_limit") < 0.0,
        }
        clean_cust, q_cust = self.delta_manager.split_and_quarantine(
            df=data["customers_dim"],
            invalid_expr_map=cust_invalid,
            quarantine_path=quarantine_path,
            batch_id=run_id,
            source_table="customers_dim",
        )
        total_quarantined += q_cust
        clean_tables["bronze_customers"] = clean_cust

        # 2. Product Catalog (no quarantine needed for 3 static rows)
        clean_tables["bronze_product_catalog"] = data["product_catalog_dim"]

        # 3. Transactions quarantine
        tx_invalid = {
            "NULL_CUSTOMER_ID": F.col("customer_id").isNull(),
            "NULL_TXN_ID": F.col("txn_id").isNull(),
            "NEGATIVE_AMOUNT": F.col("amount") < 0.0,
            "INVALID_UTILIZATION": (F.col("utilization_ratio") > 1.5)
            | (F.col("utilization_ratio") < 0.0),
        }
        clean_tx, q_tx = self.delta_manager.split_and_quarantine(
            df=data["transactions_raw"],
            invalid_expr_map=tx_invalid,
            quarantine_path=quarantine_path,
            batch_id=run_id,
            source_table="transactions_raw",
        )
        total_quarantined += q_tx
        clean_tables["bronze_transactions"] = clean_tx

        # 4. Support notes quarantine
        notes_invalid = {
            "NULL_CUSTOMER_ID": F.col("customer_id").isNull(),
            "NULL_NOTE_ID": F.col("note_id").isNull(),
        }
        clean_notes, q_notes = self.delta_manager.split_and_quarantine(
            df=data["support_notes_raw"],
            invalid_expr_map=notes_invalid,
            quarantine_path=quarantine_path,
            batch_id=run_id,
            source_table="support_notes_raw",
        )
        total_quarantined += q_notes
        clean_tables["bronze_support_notes"] = clean_notes

        # 5. Campaign labels quarantine
        labels_invalid = {
            "NULL_CUSTOMER_ID": F.col("customer_id").isNull(),
            "NULL_SNAPSHOT_MONTH": F.col("snapshot_month").isNull(),
        }
        clean_labels, q_labels = self.delta_manager.split_and_quarantine(
            df=data["campaign_labels_raw"],
            invalid_expr_map=labels_invalid,
            quarantine_path=quarantine_path,
            batch_id=run_id,
            source_table="campaign_labels_raw",
        )
        total_quarantined += q_labels
        clean_tables["bronze_campaign_labels"] = clean_labels

        # Add lineage audit columns
        enriched_tables: dict[str, DataFrame] = {}
        for tbl_name, df in clean_tables.items():
            enriched = df.withColumn("ingested_at_utc", F.current_timestamp()).withColumn(
                "source_batch_id", F.lit(run_id)
            )
            enriched_tables[tbl_name] = enriched

        return {
            "tables": enriched_tables,
            "quarantined_count": total_quarantined,
        }

    def validate(self, data: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Validate non-empty bronze datasets."""
        for tbl_name, df in data["tables"].items():
            assert df.count() > 0, f"Bronze table '{tbl_name}' is unexpectedly empty."

    def load(self, data: dict[str, Any], *args: Any, **kwargs: Any) -> StageResult:
        """Persists validated tables to Bronze Delta paths."""
        delta_root = self.config.storage.delta_dir
        delta_versions: dict[str, int] = {}
        total_output_rows = 0

        for tbl_name, df in data["tables"].items():
            path = f"{delta_root}/{tbl_name}"
            if tbl_name == "bronze_campaign_labels":
                self.delta_manager.write_overwrite_table(
                    df, path, partition_cols=["snapshot_month"]
                )
            else:
                self.delta_manager.write_overwrite_table(df, path)
            delta_versions[tbl_name] = self.delta_manager.get_latest_version(path)
            total_output_rows += df.count()

        return StageResult(
            stage_name=self.stage_name,
            run_id=kwargs.get("run_id", ""),
            status="SUCCESS",
            output_rows=total_output_rows,
            quarantine_rows=data["quarantined_count"],
            delta_versions=delta_versions,
        )
