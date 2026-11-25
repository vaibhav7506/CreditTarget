"""Governance policy evaluator implementing Model Risk (SR 11-7) pass/warn/fail gates."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from credittarget.common.artifact_store import ArtifactStore
from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig
from credittarget.exceptions import GovernanceGateError
from credittarget.logging_utils import get_logger
from credittarget.validation.fairness import compute_fairness_disparate_impact
from credittarget.validation.leakage import detect_univariate_leakage
from credittarget.validation.metrics import (
    compute_binary_classification_metrics,
    compute_csi,
    compute_decile_lift_table,
    compute_psi,
)

logger = get_logger("GovernancePolicyEvaluator")


class GovernancePolicyEvaluator:
    """Evaluates validation metrics against enterprise governance thresholds and gates promotion."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.thresholds = config.governance
        self.artifact_store = ArtifactStore(config.storage)
        self.mlflow_client = ModelRegistryClient(config.mlflow)

    def evaluate_gates(
        self,
        spark: SparkSession,
        validation_predictions_table: str | None = None,
    ) -> dict[str, Any]:
        """Runs full suite of distributed validation checks and evaluates governance gates."""
        delta_mgr = DeltaTableManager(spark)
        tbl_path = (
            validation_predictions_table
            or f"{self.config.storage.delta_dir}/model_validation_predictions"
        )
        preds_df = delta_mgr.read_table(tbl_path)

        # Separate Val (Month 9) and OOT (Months 10-12)
        val_df = preds_df.filter(preds_df["snapshot_month"].isin(self.config.splits.val_months))
        oot_df = preds_df.filter(preds_df["snapshot_month"].isin(self.config.splits.oot_months))

        # 1. Distributed Performance Metrics (Val vs OOT)
        val_prop_metrics = compute_binary_classification_metrics(
            val_df, "accepted_offer_30d", "propensity_score"
        )
        oot_prop_metrics = compute_binary_classification_metrics(
            oot_df, "accepted_offer_30d", "propensity_score"
        )

        val_pd_metrics = compute_binary_classification_metrics(val_df, "defaulted_12m", "pd_score")
        oot_pd_metrics = compute_binary_classification_metrics(oot_df, "defaulted_12m", "pd_score")

        # 2. Population Stability Index (PSI) & CSI from Val to OOT
        psi_prop, psi_prop_breakdown = compute_psi(val_df, oot_df, "propensity_score", num_bins=10)
        psi_pd, psi_pd_breakdown = compute_psi(val_df, oot_df, "pd_score", num_bins=10)

        top_numeric_features = ["spend_90d", "mean_util_90d", "credit_limit", "max_dpd_90d"]
        csi_metrics = compute_csi(val_df, oot_df, top_numeric_features)

        # 3. Decile Lift Table (OOT)
        decile_lift_table = compute_decile_lift_table(
            oot_df,
            label_col="accepted_offer_30d",
            score_col="propensity_score",
            pd_col="pd_score",
            ev_col="expected_net_value",
        )

        top2_decile_lift = 0.0
        if len(decile_lift_table) >= 2:
            r1 = decile_lift_table[0]["responders"]
            r2 = decile_lift_table[1]["responders"]
            c1 = decile_lift_table[0]["count"]
            c2 = decile_lift_table[1]["count"]
            top2_resp_rate = (r1 + r2) / (c1 + c2) if (c1 + c2) > 0 else 0.0
            total_resp = sum(d["responders"] for d in decile_lift_table)
            total_cnt = sum(d["count"] for d in decile_lift_table)
            base_rate = (total_resp / total_cnt) if total_cnt > 0 else 1.0
            top2_decile_lift = round(top2_resp_rate / base_rate, 2)

        # 4. Leakage checks
        pdf = oot_df.toPandas()
        leakage_results = detect_univariate_leakage(pdf, top_numeric_features, "accepted_offer_30d")

        # 5. Fairness checks
        fairness_results = compute_fairness_disparate_impact(pdf)
        min_di_ratio = fairness_results.get("overall_min_disparate_impact", 1.0)

        # Relative KS drop calculation
        val_ks = val_prop_metrics["ks_statistic"]
        oot_ks = oot_prop_metrics["ks_statistic"]
        ks_drop_pct = (
            round(((val_ks - oot_ks) / max(0.01, val_ks)) * 100.0, 2) if val_ks > 0 else 0.0
        )

        # Assemble unified validation metrics
        validation_metrics: dict[str, Any] = {
            "val_propensity": val_prop_metrics,
            "oot_propensity": oot_prop_metrics,
            "val_pd": val_pd_metrics,
            "oot_pd": oot_pd_metrics,
            "psi": {
                "propensity_score": psi_prop,
                "pd_score": psi_pd,
            },
            "csi": csi_metrics,
            "ks_drop_pct": ks_drop_pct,
            "top2_decile_lift": top2_decile_lift,
            "decile_lift_table": decile_lift_table,
            "leakage_violations_count": len(leakage_results),
            "disparate_impact_ratio": min_di_ratio,
            "fairness": fairness_results,
        }

        # 6. Evaluate Governance Policy Gate Rules
        gate_rules: list[dict[str, Any]] = []
        overall_status = "PASS"

        def add_rule(name: str, passed: bool, severity: str, message: str) -> None:
            nonlocal overall_status
            rule_entry = {
                "rule_name": name,
                "passed": passed,
                "severity": severity,
                "message": message,
            }
            gate_rules.append(rule_entry)
            if not passed:
                if severity == "CRITICAL":
                    overall_status = "FAIL"
                elif severity == "WARNING" and overall_status != "FAIL":
                    overall_status = "PASS_WITH_WARNINGS"

        # Rule: Critical PSI
        max_psi = max(psi_prop, psi_pd)
        add_rule(
            name="PSI_CRITICAL_THRESHOLD",
            passed=max_psi <= self.thresholds.max_psi_critical,
            severity="CRITICAL",
            message=f"Max PSI {max_psi:.4f} vs critical limit {self.thresholds.max_psi_critical:.4f}",
        )
        if (
            max_psi > self.thresholds.max_psi_warning
            and max_psi <= self.thresholds.max_psi_critical
        ):
            add_rule(
                name="PSI_WARNING_THRESHOLD",
                passed=False,
                severity="WARNING",
                message=f"PSI {max_psi:.4f} exceeds monitoring warning limit {self.thresholds.max_psi_warning:.4f}",
            )

        # Rule: KS Drop
        add_rule(
            name="MAX_KS_DROP",
            passed=ks_drop_pct <= self.thresholds.max_ks_drop_pct,
            severity="CRITICAL",
            message=f"Relative KS drop {ks_drop_pct:.2f}% vs max allowed {self.thresholds.max_ks_drop_pct:.2f}%",
        )

        # Rule: OOT Gini Propensity
        add_rule(
            name="MIN_OOT_GINI_PROPENSITY",
            passed=oot_prop_metrics["gini"] >= self.thresholds.min_oot_gini_propensity,
            severity="CRITICAL",
            message=f"OOT Gini {oot_prop_metrics['gini']:.4f} vs min {self.thresholds.min_oot_gini_propensity:.4f}",
        )

        # Rule: OOT Gini PD
        add_rule(
            name="MIN_OOT_GINI_PD",
            passed=oot_pd_metrics["gini"] >= self.thresholds.min_oot_gini_pd,
            severity="CRITICAL",
            message=f"OOT PD Gini {oot_pd_metrics['gini']:.4f} vs min {self.thresholds.min_oot_gini_pd:.4f}",
        )

        # Rule: Top-2 Decile Lift
        add_rule(
            name="MIN_TOP2_DECILE_LIFT",
            passed=top2_decile_lift >= self.thresholds.min_top2_decile_lift,
            severity="CRITICAL",
            message=f"Top-2 Decile Lift {top2_decile_lift:.2f}x vs min {self.thresholds.min_top2_decile_lift:.2f}x",
        )

        # Rule: Brier Score
        add_rule(
            name="MAX_BRIER_SCORE",
            passed=oot_prop_metrics["brier_score"] <= self.thresholds.max_brier_score,
            severity="CRITICAL",
            message=f"Brier score {oot_prop_metrics['brier_score']:.4f} vs max {self.thresholds.max_brier_score:.4f}",
        )

        # Rule: Zero Leakage Violations
        add_rule(
            name="ZERO_LEAKAGE_VIOLATIONS",
            passed=len(leakage_results) == 0,
            severity="CRITICAL",
            message=f"Detected {len(leakage_results)} leakage features",
        )

        # Rule: Disparate Impact Ratio
        add_rule(
            name="MIN_DISPARATE_IMPACT_RATIO",
            passed=min_di_ratio >= self.thresholds.min_disparate_impact_ratio,
            severity="WARNING",
            message=f"Min disparate impact ratio {min_di_ratio:.4f} vs threshold {self.thresholds.min_disparate_impact_ratio:.4f}",
        )

        gate_results: dict[str, Any] = {
            "verdict": overall_status,
            "total_rules_evaluated": len(gate_rules),
            "passed_rules": sum(1 for r in gate_rules if r["passed"]),
            "failed_rules": sum(1 for r in gate_rules if not r["passed"]),
            "rules": gate_rules,
        }

        # Persist validation artifacts
        self.artifact_store.write_json("validation/validation_metrics.json", validation_metrics)
        self.artifact_store.write_json("validation/gate_results.json", gate_results)

        # 7. Model Promotion Logic
        if overall_status in ("PASS", "PASS_WITH_WARNINGS"):
            logger.info("governance_gates_passed_promoting_models", verdict=overall_status)
            prop_version = self.mlflow_client.get_model_version_by_alias(
                self.config.mlflow.propensity_model_name, "challenger"
            )
            pd_version = self.mlflow_client.get_model_version_by_alias(
                self.config.mlflow.pd_model_name, "challenger"
            )
            self.mlflow_client.promote_to_champion(
                self.config.mlflow.propensity_model_name, prop_version
            )
            self.mlflow_client.promote_to_champion(self.config.mlflow.pd_model_name, pd_version)
        else:
            logger.error("governance_gates_failed_blocking_promotion", verdict=overall_status)
            raise GovernanceGateError(
                f"Model Risk Governance Gates FAILED. Promotion to @champion is BLOCKED.\n"
                f"Failed rules: {[r for r in gate_rules if not r['passed']]}"
            )

        return gate_results
