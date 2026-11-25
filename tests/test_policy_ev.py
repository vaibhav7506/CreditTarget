"""Tests for TargetingPolicyEngine: EV calculation, 6-tier suppression precedence, and naive comparison."""

from __future__ import annotations

import pandas as pd

from credittarget.common.artifact_store import ArtifactStore
from credittarget.config import AppConfig
from credittarget.modeling.policy import TargetingPolicyEngine


def test_ev_scalar_calculation(app_config: AppConfig) -> None:
    """Verifies expected net value scalar formula matches manual math."""
    engine = TargetingPolicyEngine(app_config)
    # P_accept = 0.20, PD = 0.05, Credit Limit = 10,000, Yield = 0.14, Fee = 60.0, EAD mult = 0.70, Promo = 15.0, LGD = 0.45
    # Revenue = 10000 * 0.14 + 60 = 1460.0
    # EAD = 10000 * 0.70 = 7000.0
    # Net Margin = (1 - 0.05) * 1460 - 0.05 * 0.45 * 7000 = 0.95 * 1460 - 157.5 = 1387.0 - 157.5 = 1229.5
    # EV = 0.20 * 1229.5 - 15.0 = 245.9 - 15.0 = 230.9
    ev = engine.compute_ev_scalar(
        propensity_score=0.20,
        pd_score=0.05,
        credit_limit=10000.0,
        base_interest_yield=0.14,
        expected_fee_revenue=60.0,
        ead_multiplier=0.70,
        promo_cost=15.0,
    )
    assert abs(ev - 230.9) < 0.01


def test_six_tier_suppression_precedence(app_config: AppConfig) -> None:
    """Verifies that the 6 suppression precedence rules evaluate in strict priority order."""
    TargetingPolicyEngine(app_config)

    # Build 6 synthetic customer rows testing each precedence rule
    rows = [
        # Customer 1: Active Delinquency (dpd >= 60) -> ACTIVE_DELINQUENCY
        {
            "customer_id": "C1",
            "propensity_score": 0.50,
            "pd_score": 0.02,
            "credit_limit": 10000.0,
            "max_dpd_90d": 65,
            "distress_flag_30d": 1,
            "max_complaint_severity_90d": 0.90,
            "base_interest_yield": 0.14,
            "expected_fee_revenue": 60.0,
            "ead_multiplier": 0.70,
            "promo_cost": 15.0,
        },
        # Customer 2: Regulatory distress -> REGULATORY_DISTRESS
        {
            "customer_id": "C2",
            "propensity_score": 0.50,
            "pd_score": 0.02,
            "credit_limit": 10000.0,
            "max_dpd_90d": 0,
            "distress_flag_30d": 1,
            "max_complaint_severity_90d": 0.85,
            "base_interest_yield": 0.14,
            "expected_fee_revenue": 60.0,
            "ead_multiplier": 0.70,
            "promo_cost": 15.0,
        },
        # Customer 3: High PD (pd > 0.08) -> PD_CAP_EXCEEDED
        {
            "customer_id": "C3",
            "propensity_score": 0.50,
            "pd_score": 0.12,
            "credit_limit": 10000.0,
            "max_dpd_90d": 0,
            "distress_flag_30d": 0,
            "max_complaint_severity_90d": 0.0,
            "base_interest_yield": 0.14,
            "expected_fee_revenue": 60.0,
            "ead_multiplier": 0.70,
            "promo_cost": 15.0,
        },
        # Customer 4: Very low limit, high promo -> NEGATIVE_EV
        {
            "customer_id": "C4",
            "propensity_score": 0.05,
            "pd_score": 0.07,
            "credit_limit": 500.0,
            "max_dpd_90d": 0,
            "distress_flag_30d": 0,
            "max_complaint_severity_90d": 0.0,
            "base_interest_yield": 0.10,
            "expected_fee_revenue": 10.0,
            "ead_multiplier": 0.70,
            "promo_cost": 25.0,  # Negative EV
        },
        # Customer 5: Eligible but ranked lower than top 30% -> BUDGET_CUTOFF
        {
            "customer_id": "C5",
            "propensity_score": 0.15,
            "pd_score": 0.03,
            "credit_limit": 3000.0,
            "max_dpd_90d": 0,
            "distress_flag_30d": 0,
            "max_complaint_severity_90d": 0.0,
            "base_interest_yield": 0.14,
            "expected_fee_revenue": 60.0,
            "ead_multiplier": 0.70,
            "promo_cost": 15.0,
        },
        # Customer 6: High EV, prime -> TARGET
        {
            "customer_id": "C6",
            "propensity_score": 0.60,
            "pd_score": 0.01,
            "credit_limit": 20000.0,
            "max_dpd_90d": 0,
            "distress_flag_30d": 0,
            "max_complaint_severity_90d": 0.0,
            "base_interest_yield": 0.14,
            "expected_fee_revenue": 60.0,
            "ead_multiplier": 0.70,
            "promo_cost": 15.0,
        },
    ]
    df = pd.DataFrame(rows)

    # Set budget fraction so only 1 row can be targeted (1/6 ~= 16%)
    app_config.policy.top_k_budget_fraction = 0.10
    engine_budget = TargetingPolicyEngine(app_config)

    res = engine_budget.evaluate_policy_pandas(df)
    c1 = res[res["customer_id"] == "C1"].iloc[0]
    c2 = res[res["customer_id"] == "C2"].iloc[0]
    c3 = res[res["customer_id"] == "C3"].iloc[0]
    c4 = res[res["customer_id"] == "C4"].iloc[0]
    c5 = res[res["customer_id"] == "C5"].iloc[0]
    c6 = res[res["customer_id"] == "C6"].iloc[0]

    assert c1["suppression_reason"] == "ACTIVE_DELINQUENCY"
    assert c2["suppression_reason"] == "REGULATORY_DISTRESS"
    assert c3["suppression_reason"] == "PD_CAP_EXCEEDED"
    assert c4["suppression_reason"] == "NEGATIVE_EV"
    assert c5["suppression_reason"] == "BUDGET_CUTOFF"
    assert c6["target_decision"] == "TARGET"
    assert c6["suppression_reason"] == "NONE"


def test_naive_vs_risk_gated_uplift(app_config: AppConfig, artifact_store: ArtifactStore) -> None:
    """Verifies that Risk-Gated EV policy produces positive net uplift and avoids credit losses."""
    engine = TargetingPolicyEngine(app_config)

    # 10 synthetic customer records with varying risk and propensity
    rows = []
    for i in range(20):
        # Customers 0-4: eager to accept (high propensity) but high default risk
        if i < 5:
            p_accept = 0.85
            pd_val = 0.25
            limit = 8000.0
            dpd = 70
        else:
            p_accept = 0.40
            pd_val = 0.02
            limit = 15000.0
            dpd = 0
        rows.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "propensity_score": p_accept,
                "pd_score": pd_val,
                "credit_limit": limit,
                "max_dpd_90d": dpd,
                "distress_flag_30d": 0,
                "max_complaint_severity_90d": 0.0,
                "base_interest_yield": 0.14,
                "expected_fee_revenue": 60.0,
                "ead_multiplier": 0.70,
                "promo_cost": 15.0,
            }
        )
    df = pd.DataFrame(rows)

    uplift = engine.compute_policy_uplift_comparison(df, artifact_store=artifact_store)
    assert uplift["avoided_credit_loss"] > 0
    assert uplift["net_ev_uplift"] > 0
    assert artifact_store.exists("validation/policy_uplift.json")
