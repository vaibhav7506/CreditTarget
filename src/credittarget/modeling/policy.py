"""Vectorized Risk-Gated Expected Net Value (EV) targeting engine and naive uplift comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credittarget.common.artifact_store import ArtifactStore
from credittarget.config import AppConfig


class TargetingPolicyEngine:
    """Evaluates Expected Net Value (EV) and applies 6-tier Model Risk suppression precedence."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.lgd = config.policy.lgd
        self.default_promo_cost = config.policy.default_promo_cost
        self.max_pd_cutoff = config.policy.max_pd_cutoff
        self.top_k_budget_fraction = config.policy.top_k_budget_fraction

    def compute_ev_scalar(
        self,
        propensity_score: float,
        pd_score: float,
        credit_limit: float,
        base_interest_yield: float = 0.14,
        expected_fee_revenue: float = 60.0,
        ead_multiplier: float = 0.70,
        promo_cost: float = 15.0,
    ) -> float:
        """Computes EV for a single customer record."""
        revenue = credit_limit * base_interest_yield + expected_fee_revenue
        ead = credit_limit * ead_multiplier
        ev = (
            propensity_score * ((1.0 - pd_score) * revenue - pd_score * self.lgd * ead) - promo_cost
        )
        return round(float(ev), 4)

    def evaluate_policy_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized evaluation of EV and 6-tier suppression precedence on a pandas DataFrame."""
        out = df.copy()

        # Extract or default commercial terms
        p_accept = out["propensity_score"].values
        pd_score = out["pd_score"].values
        credit_limit = out["credit_limit"].values
        base_yield = (
            out["base_interest_yield"].values
            if "base_interest_yield" in out
            else np.full(len(out), 0.14)
        )
        fee_rev = (
            out["expected_fee_revenue"].values
            if "expected_fee_revenue" in out
            else np.full(len(out), 60.0)
        )
        ead_mult = (
            out["ead_multiplier"].values if "ead_multiplier" in out else np.full(len(out), 0.70)
        )
        promo_cost = (
            out["promo_cost"].values
            if "promo_cost" in out
            else np.full(len(out), self.default_promo_cost)
        )

        # 1. Expected Net Value calculation
        revenue = credit_limit * base_yield + fee_rev
        ead = credit_limit * ead_mult
        net_margin = (1.0 - pd_score) * revenue - pd_score * self.lgd * ead
        expected_net_value = p_accept * net_margin - promo_cost
        out["expected_net_value"] = np.round(expected_net_value, 4)

        # Extract features for suppression rules
        max_dpd_90d = out["max_dpd_90d"].values if "max_dpd_90d" in out else np.zeros(len(out))
        distress_30d = (
            out["distress_flag_30d"].values if "distress_flag_30d" in out else np.zeros(len(out))
        )
        complaint_sev = (
            out["max_complaint_severity_90d"].values
            if "max_complaint_severity_90d" in out
            else np.zeros(len(out))
        )

        # Initial assignments
        decisions = np.array(["TARGET"] * len(out), dtype=object)
        reasons = np.array(["NONE"] * len(out), dtype=object)

        # Precedence 1: ACTIVE_DELINQUENCY (max_dpd_90d >= 60)
        cond1 = max_dpd_90d >= 60
        decisions[cond1] = "SUPPRESS"
        reasons[cond1] = "ACTIVE_DELINQUENCY"

        # Precedence 2: REGULATORY_DISTRESS (distress_flag_30d == 1 AND max_complaint_severity_90d >= 0.80)
        cond2 = (distress_30d == 1) & (complaint_sev >= 0.80) & (reasons == "NONE")
        decisions[cond2] = "SUPPRESS"
        reasons[cond2] = "REGULATORY_DISTRESS"

        # Precedence 3: PD_CAP_EXCEEDED (pd_score > max_pd_cutoff)
        cond3 = (pd_score > self.max_pd_cutoff) & (reasons == "NONE")
        decisions[cond3] = "SUPPRESS"
        reasons[cond3] = "PD_CAP_EXCEEDED"

        # Precedence 4: NEGATIVE_EV (expected_net_value <= 0.0)
        cond4 = (expected_net_value <= 0.0) & (reasons == "NONE")
        decisions[cond4] = "SUPPRESS"
        reasons[cond4] = "NEGATIVE_EV"

        out["target_decision"] = decisions
        out["suppression_reason"] = reasons

        # Precedence 5: BUDGET_CUTOFF
        # Among remaining eligible rows (reasons == "NONE"), rank descending by EV (tie-break customer_id asc)
        eligible_mask = out["suppression_reason"] == "NONE"
        total_customers = len(out)
        budget_capacity = max(1, int(np.ceil(total_customers * self.top_k_budget_fraction)))

        eligible_indices = (
            out[eligible_mask]
            .sort_values(
                by=["expected_net_value", "customer_id"],
                ascending=[False, True],
            )
            .index
        )

        if len(eligible_indices) > budget_capacity:
            suppressed_by_budget = eligible_indices[budget_capacity:]
            out.loc[suppressed_by_budget, "target_decision"] = "SUPPRESS"
            out.loc[suppressed_by_budget, "suppression_reason"] = "BUDGET_CUTOFF"

        return out

    def compute_policy_uplift_comparison(
        self,
        scored_df: pd.DataFrame,
        artifact_store: ArtifactStore | None = None,
    ) -> dict[str, Any]:
        """Compares Risk-Gated EV Policy vs. Naive Propensity-Only Targeting."""
        # 1. Risk-Gated EV Targeting results
        gated_df = self.evaluate_policy_pandas(scored_df)
        gated_targeted = gated_df[gated_df["target_decision"] == "TARGET"]

        total_customers = len(scored_df)
        budget_capacity = max(1, int(np.ceil(total_customers * self.top_k_budget_fraction)))

        # 2. Naive Propensity-Only Targeting: Top K strictly by propensity_score
        naive_top = gated_df.sort_values(
            by=["propensity_score", "customer_id"],
            ascending=[False, True],
        ).head(budget_capacity)

        # Financial & risk comparison
        gated_total_ev = float(gated_targeted["expected_net_value"].sum())
        naive_total_ev = float(naive_top["expected_net_value"].sum())
        net_ev_uplift = round(gated_total_ev - naive_total_ev, 2)

        # Expected default losses: sum(PD * LGD * EAD)
        ead_gated = gated_targeted["credit_limit"] * gated_targeted.get("ead_multiplier", 0.70)
        gated_expected_loss = float((gated_targeted["pd_score"] * self.lgd * ead_gated).sum())

        ead_naive = naive_top["credit_limit"] * naive_top.get("ead_multiplier", 0.70)
        naive_expected_loss = float((naive_top["pd_score"] * self.lgd * ead_naive).sum())
        avoided_credit_loss = round(naive_expected_loss - gated_expected_loss, 2)

        # Suppression breakdown
        suppression_counts = gated_df["suppression_reason"].value_counts().to_dict()

        uplift_report: dict[str, Any] = {
            "total_customers_evaluated": total_customers,
            "budget_capacity": budget_capacity,
            "risk_gated_targeted_count": len(gated_targeted),
            "naive_targeted_count": len(naive_top),
            "risk_gated_total_ev": round(gated_total_ev, 2),
            "naive_total_ev": round(naive_total_ev, 2),
            "net_ev_uplift": net_ev_uplift,
            "net_ev_uplift_pct": round((net_ev_uplift / max(1.0, abs(naive_total_ev))) * 100.0, 2),
            "risk_gated_expected_loss": round(gated_expected_loss, 2),
            "naive_expected_loss": round(naive_expected_loss, 2),
            "avoided_credit_loss": avoided_credit_loss,
            "suppression_breakdown": suppression_counts,
        }

        if artifact_store:
            artifact_store.write_json("validation/policy_uplift.json", uplift_report)

        return uplift_report
