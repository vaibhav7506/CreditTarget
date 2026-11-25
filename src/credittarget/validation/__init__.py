"""Distributed model validation, fairness analysis, leakage detection, and governance gates."""

from credittarget.validation.fairness import compute_fairness_disparate_impact
from credittarget.validation.gates import GovernancePolicyEvaluator
from credittarget.validation.leakage import assert_zero_leakage_columns, detect_univariate_leakage
from credittarget.validation.metrics import (
    compute_binary_classification_metrics,
    compute_csi,
    compute_decile_lift_table,
    compute_psi,
)

__all__ = [
    "GovernancePolicyEvaluator",
    "assert_zero_leakage_columns",
    "compute_binary_classification_metrics",
    "compute_csi",
    "compute_decile_lift_table",
    "compute_fairness_disparate_impact",
    "compute_psi",
    "detect_univariate_leakage",
]
