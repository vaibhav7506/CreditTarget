"""Explicit PySpark StructType contracts and canonical SHA-256 schema hasher."""

from __future__ import annotations

import hashlib
import json

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CUSTOMERS_DIM_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("region_code", StringType(), False),
        StructField("age_band", StringType(), False),
        StructField("income_tier", StringType(), False),
        StructField("tenure_months", IntegerType(), False),
        StructField("credit_limit", DoubleType(), False),
        StructField("product_segment", StringType(), False),
        StructField("account_opened_date", StringType(), False),
    ]
)

PRODUCT_CATALOG_DIM_SCHEMA = StructType(
    [
        StructField("product_segment", StringType(), False),
        StructField("base_interest_yield", DoubleType(), False),
        StructField("expected_fee_revenue", DoubleType(), False),
        StructField("ead_multiplier", DoubleType(), False),
        StructField("promo_cost", DoubleType(), False),
    ]
)

TRANSACTIONS_RAW_SCHEMA = StructType(
    [
        StructField("txn_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("txn_ts", TimestampType(), False),
        StructField("amount", DoubleType(), False),
        StructField("merchant_category", StringType(), False),
        StructField("is_delinquent_payment", IntegerType(), False),
        StructField("days_past_due", IntegerType(), False),
        StructField("utilization_ratio", DoubleType(), False),
    ]
)

SUPPORT_NOTES_RAW_SCHEMA = StructType(
    [
        StructField("note_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("note_ts", TimestampType(), False),
        StructField("channel", StringType(), False),
        StructField("note_text", StringType(), False),
    ]
)

CAMPAIGN_LABELS_RAW_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("snapshot_month", StringType(), False),
        StructField("snapshot_date", TimestampType(), False),
        StructField("accepted_offer_30d", IntegerType(), False),
        StructField("defaulted_12m", IntegerType(), False),
    ]
)

BRONZE_QUARANTINE_SCHEMA = StructType(
    [
        StructField("raw_payload", StringType(), True),
        StructField("rejection_reason", StringType(), False),
        StructField("source_table", StringType(), False),
        StructField("source_batch_id", StringType(), False),
        StructField("quarantined_at_utc", TimestampType(), False),
    ]
)

NLP_RESULT_SCHEMA = StructType(
    [
        StructField("sentiment_score", DoubleType(), True),
        StructField("distress_flag", IntegerType(), True),
        StructField("complaint_severity", DoubleType(), True),
        StructField("credit_intent_flag", IntegerType(), True),
    ]
)

GOLD_CUSTOMER_FEATURES_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("snapshot_month", StringType(), False),
        StructField("snapshot_date", TimestampType(), False),
        # Customer Demographics
        StructField("region_code", StringType(), False),
        StructField("age_band", StringType(), False),
        StructField("income_tier", StringType(), False),
        StructField("tenure_months", IntegerType(), False),
        StructField("credit_limit", DoubleType(), False),
        StructField("product_segment", StringType(), False),
        # Product Catalog Dim
        StructField("base_interest_yield", DoubleType(), False),
        StructField("expected_fee_revenue", DoubleType(), False),
        StructField("ead_multiplier", DoubleType(), False),
        StructField("promo_cost", DoubleType(), False),
        # Rolling Transaction Aggregations
        StructField("spend_30d", DoubleType(), False),
        StructField("spend_60d", DoubleType(), False),
        StructField("spend_90d", DoubleType(), False),
        StructField("txn_count_30d", IntegerType(), False),
        StructField("txn_count_90d", IntegerType(), False),
        StructField("spend_std_90d", DoubleType(), False),
        StructField("spend_velocity_ratio", DoubleType(), False),
        StructField("mean_util_30d", DoubleType(), False),
        StructField("mean_util_90d", DoubleType(), False),
        StructField("max_util_90d", DoubleType(), False),
        StructField("util_trend_30d_vs_90d", DoubleType(), False),
        StructField("max_dpd_90d", IntegerType(), False),
        StructField("dpd_events_90d", IntegerType(), False),
        StructField("dpd_events_30d", IntegerType(), False),
        StructField("severe_dpd_flag_90d", IntegerType(), False),
        # NLP Aggregations
        StructField("note_count_90d", IntegerType(), False),
        StructField("mean_sentiment_90d", DoubleType(), False),
        StructField("max_complaint_severity_90d", DoubleType(), False),
        StructField("distress_flag_30d", IntegerType(), False),
        StructField("credit_intent_flag_30d", IntegerType(), False),
        # Historical Campaign Features
        StructField("prior_campaign_contacts", IntegerType(), False),
        StructField("prior_acceptance_rate", DoubleType(), False),
        # Ground Truth Targets
        StructField("accepted_offer_30d", IntegerType(), False),
        StructField("defaulted_12m", IntegerType(), False),
    ]
)

SCORED_TARGETS_SCHEMA = StructType(
    [
        StructField("customer_id", StringType(), False),
        StructField("snapshot_month", StringType(), False),
        StructField("propensity_score", DoubleType(), False),
        StructField("pd_score", DoubleType(), False),
        StructField("expected_net_value", DoubleType(), False),
        StructField("target_decision", StringType(), False),
        StructField("suppression_reason", StringType(), False),
        StructField("propensity_model_version", StringType(), False),
        StructField("pd_model_version", StringType(), False),
        StructField("gold_delta_version", LongType(), False),
        StructField("feature_schema_hash", StringType(), False),
        StructField("scored_at_utc", TimestampType(), False),
    ]
)


def compute_schema_hash(schema: StructType) -> str:
    """Computes a deterministic canonical SHA-256 hash for a PySpark StructType."""
    fields_repr = []
    for field in sorted(schema.fields, key=lambda f: f.name):
        fields_repr.append(
            {
                "name": field.name,
                "dataType": field.dataType.simpleString(),
                "nullable": field.nullable,
            }
        )
    canonical_json = json.dumps(fields_repr, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
