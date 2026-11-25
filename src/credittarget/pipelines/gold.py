"""Gold Lakehouse stage: point-in-time joins, rolling SQL windows, BroadcastHashJoin, and feature store assembly."""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window

from credittarget.common.artifact_store import ArtifactStore
from credittarget.common.base_stage import BasePipelineStage, StageResult
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.dq import DataQualityChecker
from credittarget.common.spark_utils import export_physical_plan, safe_broadcast_join
from credittarget.config import AppConfig


class GoldStage(BasePipelineStage):
    """Gold Stage constructs the customer ML feature store with strict point-in-time and broadcast joins."""

    def __init__(self, spark: SparkSession, config: AppConfig) -> None:
        super().__init__(config=config, stage_name="GoldStage")
        self.spark = spark
        self.delta_manager = DeltaTableManager(spark)
        self.artifact_store = ArtifactStore(config.storage)

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, DataFrame]:
        """Extracts required silver tables."""
        delta_root = self.config.storage.delta_dir
        return {
            "silver_customers": self.delta_manager.read_table(f"{delta_root}/silver_customers"),
            "silver_product_catalog": self.delta_manager.read_table(
                f"{delta_root}/silver_product_catalog"
            ),
            "silver_transactions": self.delta_manager.read_table(
                f"{delta_root}/silver_transactions"
            ),
            "silver_support_notes_nlp": self.delta_manager.read_table(
                f"{delta_root}/silver_support_notes_nlp"
            ),
            "silver_campaign_labels": self.delta_manager.read_table(
                f"{delta_root}/silver_campaign_labels"
            ),
        }

    def transform(self, data: dict[str, DataFrame], *args: Any, **kwargs: Any) -> DataFrame:
        """Executes point-in-time joins, rolling window aggregations, and broadcast dimension joins."""
        campaign = data["silver_campaign_labels"]
        tx = data["silver_transactions"]
        notes = data["silver_support_notes_nlp"]
        customers = data["silver_customers"]
        catalog = data["silver_product_catalog"]

        # 1. Point-in-time join with transactions (90-day lookback: snapshot_date - 90d < txn_ts <= snapshot_date)
        tx_renamed = tx.withColumnRenamed("customer_id", "_tx_customer_id")
        tx_cond = (
            (campaign.customer_id == tx_renamed._tx_customer_id)
            & (tx_renamed.txn_ts <= campaign.snapshot_date)
            & (tx_renamed.txn_ts > campaign.snapshot_date - F.expr("INTERVAL 90 DAYS"))
        )
        tx_joined = campaign.select(
            campaign.customer_id,
            campaign.snapshot_month,
            campaign.snapshot_date,
        ).join(tx_renamed, tx_cond, "inner")

        # Verify point-in-time invariant on non-empty join
        if tx_joined.count() > 0:
            DataQualityChecker.assert_point_in_time(
                tx_joined, "txn_ts", "snapshot_date", "tx_pit_join"
            )

        # Aggregate transaction features per (customer_id, snapshot_month)
        tx_aggs = (
            tx_joined.groupBy("customer_id", "snapshot_month")
            .agg(
                F.coalesce(
                    F.sum(
                        F.when(
                            F.col("txn_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"),
                            F.col("amount"),
                        )
                    ),
                    F.lit(0.0),
                ).alias("spend_30d"),
                F.coalesce(
                    F.sum(
                        F.when(
                            F.col("txn_ts") > F.col("snapshot_date") - F.expr("INTERVAL 60 DAYS"),
                            F.col("amount"),
                        )
                    ),
                    F.lit(0.0),
                ).alias("spend_60d"),
                F.coalesce(F.sum(F.col("amount")), F.lit(0.0)).alias("spend_90d"),
                F.coalesce(
                    F.count(
                        F.when(
                            F.col("txn_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"), 1
                        )
                    ),
                    F.lit(0),
                ).alias("txn_count_30d"),
                F.coalesce(F.count(F.col("txn_id")), F.lit(0)).alias("txn_count_90d"),
                F.coalesce(F.stddev(F.col("amount")), F.lit(0.0)).alias("spend_std_90d"),
                F.coalesce(
                    F.avg(
                        F.when(
                            F.col("txn_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"),
                            F.col("utilization_ratio"),
                        )
                    ),
                    F.lit(0.0),
                ).alias("mean_util_30d"),
                F.coalesce(F.avg(F.col("utilization_ratio")), F.lit(0.0)).alias("mean_util_90d"),
                F.coalesce(F.max(F.col("utilization_ratio")), F.lit(0.0)).alias("max_util_90d"),
                F.coalesce(F.max(F.col("days_past_due")), F.lit(0)).alias("max_dpd_90d"),
                F.coalesce(F.sum(F.when(F.col("days_past_due") > 0, 1)), F.lit(0)).alias(
                    "dpd_events_90d"
                ),
                F.coalesce(
                    F.sum(
                        F.when(
                            (F.col("txn_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"))
                            & (F.col("days_past_due") > 0),
                            1,
                        )
                    ),
                    F.lit(0),
                ).alias("dpd_events_30d"),
            )
            .withColumn(
                "spend_velocity_ratio", F.col("spend_30d") / (F.col("spend_90d") / 3.0 + 1e-6)
            )
            .withColumn("util_trend_30d_vs_90d", F.col("mean_util_30d") - F.col("mean_util_90d"))
            .withColumn(
                "severe_dpd_flag_90d",
                F.when(F.col("max_dpd_90d") >= 60, F.lit(1)).otherwise(F.lit(0)),
            )
        )

        # 2. Point-in-time join with support notes (90-day lookback)
        notes_renamed = notes.withColumnRenamed("customer_id", "_notes_customer_id")
        notes_cond = (
            (campaign.customer_id == notes_renamed._notes_customer_id)
            & (notes_renamed.note_ts <= campaign.snapshot_date)
            & (notes_renamed.note_ts > campaign.snapshot_date - F.expr("INTERVAL 90 DAYS"))
        )
        notes_joined = campaign.select(
            campaign.customer_id,
            campaign.snapshot_month,
            campaign.snapshot_date,
        ).join(notes_renamed, notes_cond, "inner")

        if notes_joined.count() > 0:
            DataQualityChecker.assert_point_in_time(
                notes_joined, "note_ts", "snapshot_date", "notes_pit_join"
            )

        # Aggregate NLP features per (customer_id, snapshot_month)
        notes_aggs = notes_joined.groupBy("customer_id", "snapshot_month").agg(
            F.coalesce(F.count(F.col("note_id")), F.lit(0)).alias("note_count_90d"),
            F.coalesce(F.avg(F.col("sentiment_score")), F.lit(0.0)).alias("mean_sentiment_90d"),
            F.coalesce(F.max(F.col("complaint_severity")), F.lit(0.0)).alias(
                "max_complaint_severity_90d"
            ),
            F.coalesce(
                F.max(
                    F.when(
                        F.col("note_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"),
                        F.col("distress_flag"),
                    )
                ),
                F.lit(0),
            ).alias("distress_flag_30d"),
            F.coalesce(
                F.max(
                    F.when(
                        F.col("note_ts") > F.col("snapshot_date") - F.expr("INTERVAL 30 DAYS"),
                        F.col("credit_intent_flag"),
                    )
                ),
                F.lit(0),
            ).alias("credit_intent_flag_30d"),
        )

        # 3. Assemble customer features with historical leak-free campaign window
        base_features = campaign.join(
            tx_aggs, on=["customer_id", "snapshot_month"], how="left"
        ).join(notes_aggs, on=["customer_id", "snapshot_month"], how="left")

        # Fill default zeros for customers with no transactions or notes in the 90d window
        base_features = base_features.na.fill(
            {
                "spend_30d": 0.0,
                "spend_60d": 0.0,
                "spend_90d": 0.0,
                "txn_count_30d": 0,
                "txn_count_90d": 0,
                "spend_std_90d": 0.0,
                "spend_velocity_ratio": 0.0,
                "mean_util_30d": 0.0,
                "mean_util_90d": 0.0,
                "max_util_90d": 0.0,
                "util_trend_30d_vs_90d": 0.0,
                "max_dpd_90d": 0,
                "dpd_events_90d": 0,
                "dpd_events_30d": 0,
                "severe_dpd_flag_90d": 0,
                "note_count_90d": 0,
                "mean_sentiment_90d": 0.0,
                "max_complaint_severity_90d": 0.0,
                "distress_flag_30d": 0,
                "credit_intent_flag_30d": 0,
            }
        )

        # Strictly leak-free historical campaign features: unboundedPreceding to -1
        w_hist = (
            Window.partitionBy("customer_id")
            .orderBy("snapshot_month")
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        base_features = base_features.withColumn(
            "prior_campaign_contacts",
            F.coalesce(F.count("accepted_offer_30d").over(w_hist), F.lit(0)),
        ).withColumn(
            "prior_acceptance_rate",
            F.coalesce(F.avg("accepted_offer_30d").over(w_hist), F.lit(0.0)),
        )

        # 4. Join customer demographics
        cust_clean = customers.drop("ingested_at_utc", "source_batch_id")
        gold_with_cust = base_features.join(cust_clean, on="customer_id", how="inner")

        # 5. Broadcast join product catalog dimension
        cat_clean = catalog.drop("ingested_at_utc", "source_batch_id")
        gold_df = safe_broadcast_join(
            fact_df=gold_with_cust,
            dim_df=cat_clean,
            on="product_segment",
            how="left",
        )

        from credittarget.schemas.contracts import GOLD_CUSTOMER_FEATURES_SCHEMA

        gold_cols = [f.name for f in GOLD_CUSTOMER_FEATURES_SCHEMA.fields]
        return gold_df.select(*gold_cols)

    def validate(self, data: DataFrame, *args: Any, **kwargs: Any) -> None:
        """Runs DQ validations on composite keys, nulls, and numeric bounds."""
        DataQualityChecker.assert_non_empty(data, "gold_customer_features")
        DataQualityChecker.assert_unique_keys(
            data, ["customer_id", "snapshot_month"], "gold_customer_features"
        )

        # Null bounds on essential model features
        DataQualityChecker.assert_max_null_rate(
            data,
            {
                "customer_id": 0.0,
                "snapshot_month": 0.0,
                "credit_limit": 0.0,
                "spend_90d": 0.0,
                "mean_util_90d": 0.0,
                "accepted_offer_30d": 0.0,
                "defaulted_12m": 0.0,
            },
            "gold_customer_features",
        )

    def load(self, data: DataFrame, *args: Any, **kwargs: Any) -> StageResult:
        """Exports physical plan with BroadcastHashJoin verification and writes Gold Delta table."""
        delta_root = self.config.storage.delta_dir
        gold_path = f"{delta_root}/gold_customer_features"

        # Verify BroadcastHashJoin operator in Spark execution plan
        export_physical_plan(
            data,
            artifact_store=self.artifact_store,
            rel_path="spark_plans/gold_physical_plan.txt",
            required_operators=["BroadcastHashJoin"],
        )

        # Write partitioned by snapshot_month
        self.delta_manager.write_overwrite_table(data, gold_path, partition_cols=["snapshot_month"])
        latest_version = self.delta_manager.get_latest_version(gold_path)

        return StageResult(
            stage_name=self.stage_name,
            run_id=kwargs.get("run_id", ""),
            status="SUCCESS",
            output_rows=data.count(),
            delta_versions={"gold_customer_features": latest_version},
            metadata={"gold_delta_version": latest_version},
        )
