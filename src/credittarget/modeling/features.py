"""Feature definitions, scikit-learn ColumnTransformer preprocessor, and leakage assertions."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credittarget.exceptions import LeakageDetectedError

NUMERIC_FEATURES = [
    "tenure_months",
    "credit_limit",
    "spend_30d",
    "spend_60d",
    "spend_90d",
    "txn_count_30d",
    "txn_count_90d",
    "spend_std_90d",
    "spend_velocity_ratio",
    "mean_util_30d",
    "mean_util_90d",
    "max_util_90d",
    "util_trend_30d_vs_90d",
    "max_dpd_90d",
    "dpd_events_90d",
    "dpd_events_30d",
    "severe_dpd_flag_90d",
    "note_count_90d",
    "mean_sentiment_90d",
    "max_complaint_severity_90d",
    "distress_flag_30d",
    "credit_intent_flag_30d",
    "prior_campaign_contacts",
    "prior_acceptance_rate",
    "base_interest_yield",
    "expected_fee_revenue",
    "ead_multiplier",
    "promo_cost",
]

CATEGORICAL_FEATURES = [
    "region_code",
    "age_band",
    "income_tier",
    "product_segment",
]

FORBIDDEN_LEAKAGE_COLUMNS = [
    "accepted_offer_30d",
    "defaulted_12m",
    "snapshot_date",
    "ingested_at_utc",
    "source_batch_id",
    "quarantined_at_utc",
    "rejection_reason",
]


def validate_feature_columns(
    feature_names: list[str],
    forbidden_columns: list[str] | None = None,
) -> None:
    """Asserts that no forbidden or leakage columns appear in the feature specification."""
    forbidden = set(forbidden_columns or FORBIDDEN_LEAKAGE_COLUMNS)
    leaked = [col for col in feature_names if col in forbidden]
    if leaked:
        raise LeakageDetectedError(
            f"Forbidden leakage columns found in modeling features: {leaked}"
        )


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer preprocessor fit strictly on training splits."""
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    validate_feature_columns(num_cols)
    validate_feature_columns(cat_cols)

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor
