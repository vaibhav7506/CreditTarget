"""Tests for Grounded GenAI Validation Memo reporter: reconciliation and Jinja2 fallback."""

from __future__ import annotations

import pytest

from credittarget.common.artifact_store import ArtifactStore
from credittarget.config import AppConfig
from credittarget.exceptions import ReconciliationError
from credittarget.governance.reporter import (
    ModelRiskValidationReporter,
    reconcile_memo_numbers,
)


def test_reconcile_memo_numbers_grounded() -> None:
    """Verifies that a memo referencing only verified artifact numbers passes reconciliation."""
    source_artifacts = {
        "metrics": {"roc_auc": 0.785, "gini": 0.57, "ks": 41.5},
        "uplift": {"net_ev": 15420.50, "loss_avoided": 8900.25},
    }

    memo = (
        "The model achieved an OOT ROC-AUC of 0.785 with a Gini coefficient of 0.57. "
        "The KS statistic was 41.5% and total net EV was $15420.50 with avoided losses of $8900.25."
    )

    # Should pass without error
    reconcile_memo_numbers(memo, source_artifacts)


def test_reconcile_memo_numbers_detects_hallucination() -> None:
    """Verifies that hallucinated numbers trigger a ReconciliationError."""
    source_artifacts = {
        "metrics": {"roc_auc": 0.785, "gini": 0.57},
    }

    # 0.999 is hallucinated
    memo = "The model achieved an unprecedented ROC-AUC of 0.999 and Gini of 0.57."

    with pytest.raises(ReconciliationError, match="detected ungrounded/hallucinated numbers"):
        reconcile_memo_numbers(memo, source_artifacts)


def test_reporter_end_to_end_generation(
    app_config: AppConfig, artifact_store: ArtifactStore
) -> None:
    """Verifies end-to-end memo rendering from written JSON artifacts."""
    # Write mock artifacts
    artifact_store.write_json(
        "validation/gate_results.json",
        {
            "verdict": "PASS",
            "passed_rules": 7,
            "total_rules_evaluated": 7,
        },
    )
    artifact_store.write_json(
        "validation/validation_metrics.json",
        {
            "oot_propensity": {
                "roc_auc": 0.78,
                "gini": 0.56,
                "ks_statistic": 42.0,
                "brier_score": 0.12,
            },
            "oot_pd": {"roc_auc": 0.81, "gini": 0.62, "ks_statistic": 45.0, "brier_score": 0.08},
            "psi": {"propensity_score": 0.05, "pd_score": 0.04},
            "csi": {"spend_90d": 0.02},
            "top2_decile_lift": 2.25,
            "disparate_impact_ratio": 0.85,
            "fairness": {
                "region_code": {"disparate_impact_ratio": 0.88},
                "age_band": {"disparate_impact_ratio": 0.85},
            },
            "decile_lift_table": [
                {
                    "decile": 1,
                    "count": 100,
                    "responders": 35,
                    "response_rate": 0.35,
                    "lift": 2.5,
                    "cumulative_capture_rate": 0.35,
                    "mean_propensity": 0.75,
                    "avg_pd": 0.02,
                    "cumulative_ev": 5000.0,
                }
            ],
        },
    )
    artifact_store.write_json(
        "validation/policy_uplift.json",
        {
            "budget_capacity": 600,
            "risk_gated_total_ev": 45000.0,
            "naive_total_ev": 32000.0,
            "net_ev_uplift": 13000.0,
            "net_ev_uplift_pct": 40.6,
            "avoided_credit_loss": 12500.0,
            "suppression_breakdown": {"BUDGET_CUTOFF": 300, "NEGATIVE_EV": 50},
        },
    )
    artifact_store.write_json(
        "governance/lineage.json",
        {
            "generated_at_utc": "2026-11-24T12:00:00Z",
            "git_sha": "abc1234",
            "gold_delta_version": 1,
            "propensity_model_version": "1",
            "pd_model_version": "1",
            "storage_backend": "local",
        },
    )

    reporter = ModelRiskValidationReporter(app_config)
    memo = reporter.generate_memo()
    assert "# Model Risk Validation Memo: CreditTarget Engine" in memo
    assert artifact_store.exists("governance/model_risk_validation_memo.md")
