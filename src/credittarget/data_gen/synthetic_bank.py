"""Pure PySpark scalable synthetic banking data generator across 5 relational tables."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from credittarget.config import AppConfig
from credittarget.schemas.contracts import (
    PRODUCT_CATALOG_DIM_SCHEMA,
    TRANSACTIONS_RAW_SCHEMA,
)

NOTE_TEMPLATES = [
    "Customer requested credit limit increase due to upcoming travel and good credit history",
    "Inquired about reward points exchange rates and international travel perks on premier card",
    "Customer dispute: billing fee charged on statement is completely unacceptable, demanding refund",
    "Hardship notice: customer experienced medical emergency, unable to pay minimum due on time",
    "Threatening to close account due to unsatisfactory customer service and high annual fees",
    "Customer called to verify recent billing statement balance and payment due date",
]


def generate_synthetic_bank_data(
    spark: SparkSession,
    config: AppConfig,
) -> dict[str, DataFrame]:
    """Generates 5 synthetic banking tables purely using scalable Spark expressions."""
    num_customers = config.datagen.num_customers

    # 1. Product Catalog Dimension (3 rows)
    catalog_data = [
        ("BASIC", 0.18, 25.0, 0.75, 10.0),
        ("PLUS", 0.14, 60.0, 0.70, 15.0),
        ("PREMIER", 0.10, 150.0, 0.60, 25.0),
    ]
    df_catalog = spark.createDataFrame(catalog_data, schema=PRODUCT_CATALOG_DIM_SCHEMA)

    # 2. Customers Dimension
    cust_base = spark.range(num_customers).toDF("id")
    regions = ["NORTH", "SOUTH", "EAST", "WEST"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    income_tiers = ["LOW", "MID", "HIGH", "AFFLUENT"]
    segments = ["BASIC", "PLUS", "PREMIER"]

    df_customers = (
        cust_base.withColumn(
            "customer_id", F.concat(F.lit("CUST_"), F.lpad(F.col("id").cast("string"), 7, "0"))
        )
        .withColumn(
            "region_code",
            F.element_at(F.array([F.lit(r) for r in regions]), (F.col("id") % 4 + 1).cast("int")),
        )
        .withColumn(
            "age_band",
            F.element_at(
                F.array([F.lit(a) for a in age_bands]), ((F.col("id") / 3) % 4 + 1).cast("int")
            ),
        )
        .withColumn(
            "income_tier",
            F.element_at(
                F.array([F.lit(i) for i in income_tiers]), ((F.col("id") / 5) % 4 + 1).cast("int")
            ),
        )
        .withColumn("tenure_months", (F.col("id") % 115 + 6).cast("int"))
        .withColumn(
            "product_segment",
            F.element_at(
                F.array([F.lit(s) for s in segments]), ((F.col("id") / 7) % 3 + 1).cast("int")
            ),
        )
        .withColumn(
            "credit_limit",
            F.when(F.col("income_tier") == "LOW", 2500.0)
            .when(F.col("income_tier") == "MID", 6000.0)
            .when(F.col("income_tier") == "HIGH", 14000.0)
            .otherwise(25000.0),
        )
        .withColumn(
            "account_opened_date",
            F.date_sub(F.lit("2025-01-01"), (F.col("tenure_months") * 30).cast("int")).cast(
                "string"
            ),
        )
        .select(
            "customer_id",
            "region_code",
            "age_band",
            "income_tier",
            "tenure_months",
            "credit_limit",
            "product_segment",
            "account_opened_date",
        )
    )

    # 3. Transactions Raw (~70 txns per customer across 2024-10-01 to 2025-12-31, 456 days)
    tx_per_customer = 70
    df_tx_seq = spark.range(tx_per_customer).toDF("tx_seq")
    df_tx_grid = cust_base.crossJoin(df_tx_seq)

    categories = ["GROCERY", "DINING", "TRAVEL", "ELECTRONICS", "UTILITIES", "ENTERTAINMENT"]

    # Calculate day offset: spans 0..455 days from 2024-10-01
    day_offset = ((F.col("id") * 37 + F.col("tx_seq") * 13) % 456).cast("int")
    sec_offset = ((F.col("id") * 101 + F.col("tx_seq") * 313) % 86400).cast("int")

    # Base utilization & delinquency factors correlated with customer risk
    cust_base_risk = ((F.col("id") * 17) % 100).cast("double") / 100.0
    tx_noise = (((F.col("tx_seq") * 7) % 21).cast("double") - 10.0) / 100.0
    risk_factor = F.least(F.lit(1.0), F.greatest(F.lit(0.0), cust_base_risk + tx_noise))

    # Mild covariate shift in late months (Nov-Dec 2025: days 396-456)
    shift_adder = F.when(day_offset >= 396, F.lit(0.02)).otherwise(F.lit(0.0))

    util_expr = F.least(
        F.lit(0.95),
        F.greatest(F.lit(0.05), (0.15 + risk_factor * 0.65 + shift_adder)),
    )

    # Delinquency logic: risk_factor > 0.85 generates dpd
    is_delinq_expr = F.when(risk_factor > 0.85, F.lit(1)).otherwise(F.lit(0))
    dpd_expr = (
        F.when(
            risk_factor > 0.94,
            ((risk_factor - 0.94) * 1500 + 30).cast("int"),
        )
        .when(
            risk_factor > 0.85,
            ((risk_factor - 0.85) * 350 + 1).cast("int"),
        )
        .otherwise(F.lit(0))
    )

    df_tx_valid = (
        df_tx_grid.withColumn(
            "txn_id",
            F.concat(
                F.lit("TXN_"),
                F.lpad((F.col("id") * 1000 + F.col("tx_seq")).cast("string"), 10, "0"),
            ),
        )
        .withColumn(
            "customer_id", F.concat(F.lit("CUST_"), F.lpad(F.col("id").cast("string"), 7, "0"))
        )
        .withColumn(
            "txn_ts",
            F.to_timestamp(F.date_add(F.lit("2024-10-01"), day_offset))
            + F.expr("INTERVAL 1 SECOND") * sec_offset,
        )
        .withColumn(
            "amount",
            F.round(((F.col("id") * 29 + F.col("tx_seq") * 53) % 450 + 15).cast("double"), 2),
        )
        .withColumn(
            "merchant_category",
            F.element_at(
                F.array([F.lit(c) for c in categories]),
                ((F.col("id") + F.col("tx_seq")) % 6 + 1).cast("int"),
            ),
        )
        .withColumn("is_delinquent_payment", is_delinq_expr)
        .withColumn("days_past_due", dpd_expr)
        .withColumn("utilization_ratio", F.round(util_expr, 4))
        .select(
            "txn_id",
            "customer_id",
            "txn_ts",
            "amount",
            "merchant_category",
            "is_delinquent_payment",
            "days_past_due",
            "utilization_ratio",
        )
    )

    # Inject ~1% duplicate transactions
    tx_count = num_customers * tx_per_customer
    dup_tx = df_tx_valid.limit(max(1, int(tx_count * 0.01)))

    # Inject ~0.5% invalid records (negative utilization or null customer_id)
    StructType(
        [
            StructField("txn_id", StringType(), False),
            StructField("customer_id", StringType(), True),
            StructField("txn_ts", TRANSACTIONS_RAW_SCHEMA["txn_ts"].dataType, False),
            StructField("amount", DoubleType(), False),
            StructField("merchant_category", StringType(), False),
            StructField("is_delinquent_payment", DoubleType(), False),
            StructField("days_past_due", DoubleType(), False),
            StructField("utilization_ratio", DoubleType(), False),
        ]
    )
    invalid_tx = (
        df_tx_valid.limit(max(1, int(tx_count * 0.005)))
        .withColumn("txn_id", F.concat(F.lit("INV_"), F.col("txn_id")))
        .withColumn("utilization_ratio", F.lit(1.99))  # Bound violation (> 1.5)
    )

    df_transactions = df_tx_valid.union(dup_tx).union(invalid_tx)

    # 4. Support Notes Raw (~9 notes per customer over 456 days)
    notes_per_customer = 9
    df_note_seq = spark.range(notes_per_customer).toDF("note_seq")
    df_notes_grid = cust_base.crossJoin(df_note_seq)

    channels = ["CALL_CENTER", "CHAT", "EMAIL"]
    note_day_offset = ((F.col("id") * 43 + F.col("note_seq") * 47) % 456).cast("int")
    note_sec_offset = ((F.col("id") * 137 + F.col("note_seq") * 257) % 86400).cast("int")

    df_notes_valid = (
        df_notes_grid.withColumn(
            "note_id",
            F.concat(
                F.lit("NOTE_"),
                F.lpad((F.col("id") * 100 + F.col("note_seq")).cast("string"), 9, "0"),
            ),
        )
        .withColumn(
            "customer_id", F.concat(F.lit("CUST_"), F.lpad(F.col("id").cast("string"), 7, "0"))
        )
        .withColumn(
            "note_ts",
            F.to_timestamp(F.date_add(F.lit("2024-10-01"), note_day_offset))
            + F.expr("INTERVAL 1 SECOND") * note_sec_offset,
        )
        .withColumn(
            "channel",
            F.element_at(
                F.array([F.lit(ch) for ch in channels]),
                ((F.col("id") + F.col("note_seq")) % 3 + 1).cast("int"),
            ),
        )
        .withColumn(
            "note_text",
            F.element_at(
                F.array([F.lit(t) for t in NOTE_TEMPLATES]),
                ((F.col("id") * 3 + F.col("note_seq") * 5) % 6 + 1).cast("int"),
            ),
        )
        .select("note_id", "customer_id", "note_ts", "channel", "note_text")
    )

    # Inject ~1% duplicate notes and ~0.5% invalid notes
    note_count = num_customers * notes_per_customer
    dup_notes = df_notes_valid.limit(max(1, int(note_count * 0.01)))
    invalid_notes = (
        df_notes_valid.limit(max(1, int(note_count * 0.005)))
        .withColumn("note_id", F.concat(F.lit("INV_"), F.col("note_id")))
        .withColumn("customer_id", F.lit(None).cast("string"))  # Null PK violation
    )
    df_notes = df_notes_valid.union(dup_notes).union(invalid_notes)

    # 5. Campaign Labels Raw (12 monthly snapshots: 2025-01 through 2025-12)
    snapshot_months = [
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
    ]
    month_data = [(m, f"{m}-01 00:00:00") for m in snapshot_months]
    df_months = spark.createDataFrame(month_data, ["snapshot_month", "snapshot_date_str"])

    # Latent probability model using strictly pre-snapshot customer latent properties
    df_campaign_grid = cust_base.crossJoin(df_months)

    # Latent features derived from id and month
    m_idx = F.substring(F.col("snapshot_month"), 6, 2).cast("int")
    cust_seed = ((F.col("id") * 13 + m_idx * 17) % 1000).cast("double") / 1000.0
    cust_credit_intent = ((F.col("id") * 31 + m_idx * 7) % 100).cast("double") / 100.0
    cust_risk = ((F.col("id") * 17) % 100).cast("double") / 100.0

    # Propensity latent log-odds: targets ~15% base rate
    # Higher for premier/plus, moderate tenure, credit intent notes
    propensity_logit = (
        -1.8
        + 1.2 * cust_credit_intent
        + 0.5 * (1.0 - cust_risk)
        + 0.3 * F.sin(m_idx.cast("double") * 0.5)
    )
    propensity_p = 1.0 / (1.0 + F.exp(-propensity_logit))
    accepted_offer = F.when(cust_seed < propensity_p, F.lit(1)).otherwise(F.lit(0))

    # Default latent log-odds: targets ~6-8% base rate
    # Strongly aligned with customer risk factor that drives delinquency and utilization
    default_logit = -4.5 + 5.5 * cust_risk
    default_p = 1.0 / (1.0 + F.exp(-default_logit))
    default_seed = ((F.col("id") * 59 + m_idx * 17) % 1000).cast("double") / 1000.0
    defaulted = F.when(default_seed < default_p, F.lit(1)).otherwise(F.lit(0))

    df_campaign = (
        df_campaign_grid.withColumn(
            "customer_id", F.concat(F.lit("CUST_"), F.lpad(F.col("id").cast("string"), 7, "0"))
        )
        .withColumn("snapshot_date", F.to_timestamp(F.col("snapshot_date_str")))
        .withColumn("accepted_offer_30d", accepted_offer.cast("int"))
        .withColumn("defaulted_12m", defaulted.cast("int"))
        .select(
            "customer_id",
            "snapshot_month",
            "snapshot_date",
            "accepted_offer_30d",
            "defaulted_12m",
        )
    )

    return {
        "customers_dim": df_customers,
        "product_catalog_dim": df_catalog,
        "transactions_raw": df_transactions,
        "support_notes_raw": df_notes,
        "campaign_labels_raw": df_campaign,
    }
