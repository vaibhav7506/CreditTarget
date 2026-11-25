"""Tests for GovernancePolicyEvaluator: PASS promotion and FAIL blocking on breach."""

from __future__ import annotations

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from credittarget.common.delta_io import DeltaTableManager
from credittarget.common.mlflow_client import ModelRegistryClient
from credittarget.config import AppConfig
from credittarget.exceptions import GovernanceGateError
from credittarget.validation.gates import GovernancePolicyEvaluator


def create_mock_predictions_table(
    spark: SparkSession,
    table_path: str,
    psi_breach: bool = False,
    poor_gini: bool = False,
) -> None:
    """Creates a mock model_validation_predictions Delta table for testing gates."""
    delta_mgr = DeltaTableManager(spark)

    # Months: 2025-09 (Val), 2025-10, 2025-11, 2025-12 (OOT)
    rows = []
    for month in ["2025-09", "2025-10", "2025-11", "2025-12"]:
        is_val = month == "2025-09"
        for i in range(200):
            # Normal scores
            if psi_breach and not is_val:
                p_score = 0.95  # Shifted to high values in OOT
            else:
                p_score = (i % 100) / 100.0

            if poor_gini:
                label = i % 2  # Random label -> Gini ~0
            else:
                label = 1 if p_score >= 0.50 else 0

            rows.append(
                {
                    "customer_id": f"C_{month}_{i:03d}",
                    "snapshot_month": month,
                    "propensity_score": float(p_score),
                    "pd_score": float(0.02 if label == 0 else 0.05),
                    "accepted_offer_30d": int(label),
                    "defaulted_12m": int(label),
                    "expected_net_value": float(p_score * 100.0),
                    "spend_90d": float(100.0 * (i + 1)),
                    "mean_util_90d": float((i % 50) / 100.0),
                    "credit_limit": 5000.0,
                    "max_dpd_90d": 0,
                    "region_code": "NORTH" if i % 2 == 0 else "SOUTH",
                    "age_band": "26-35" if i % 2 == 0 else "36-50",
                    "target_decision": "TARGET" if p_score >= 0.5 else "SUPPRESS",
                }
            )

    pdf = pd.DataFrame(rows)
    sdf = spark.createDataFrame(pdf)
    delta_mgr.write_overwrite_table(sdf, table_path, partition_cols=["snapshot_month"])


def test_governance_gate_pass_promotes_champion(
    spark: SparkSession,
    app_config: AppConfig,
    mlflow_client: ModelRegistryClient,
) -> None:
    """Verifies that clean metrics pass governance gates and promote challenger to champion."""
    tbl_path = f"{app_config.storage.delta_dir}/test_valid_predictions"
    create_mock_predictions_table(spark, tbl_path, psi_breach=False, poor_gini=False)

    # Register initial dummy models with @challenger alias
    from sklearn.linear_model import LogisticRegression

    dummy = LogisticRegression()
    dummy.fit([[0.0], [1.0]], [0, 1])

    with mlflow_client.start_run(run_name="init_models"):
        mlflow_client.log_model_and_register(
            dummy, "p_dummy", app_config.mlflow.propensity_model_name, initial_alias="challenger"
        )
        mlflow_client.log_model_and_register(
            dummy, "pd_dummy", app_config.mlflow.pd_model_name, initial_alias="challenger"
        )

    evaluator = GovernancePolicyEvaluator(app_config)
    res = evaluator.evaluate_gates(spark, validation_predictions_table=tbl_path)

    assert res["verdict"] in ("PASS", "PASS_WITH_WARNINGS")

    # Verify model was promoted to champion
    champ_version = mlflow_client.get_model_version_by_alias(
        app_config.mlflow.propensity_model_name, "champion"
    )
    assert champ_version is not None


def test_governance_gate_fail_blocks_promotion(
    spark: SparkSession,
    app_config: AppConfig,
) -> None:
    """Verifies that a critical failure (e.g., severe PSI breach or Gini collapse) blocks promotion."""
    tbl_path = f"{app_config.storage.delta_dir}/test_fail_predictions"
    create_mock_predictions_table(spark, tbl_path, psi_breach=True, poor_gini=True)

    evaluator = GovernancePolicyEvaluator(app_config)
    with pytest.raises(GovernanceGateError, match="Model Risk Governance Gates FAILED"):
        evaluator.evaluate_gates(spark, validation_predictions_table=tbl_path)
