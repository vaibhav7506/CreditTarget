"""Machine learning modeling, calibration, features, and EV policy targeting package."""

from credittarget.modeling.calibration import PlattCalibratedClassifier
from credittarget.modeling.features import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    validate_feature_columns,
)
from credittarget.modeling.policy import TargetingPolicyEngine
from credittarget.modeling.splits import get_temporal_splits, validate_temporal_ordering

try:
    from credittarget.modeling.train import train_all_models
except (ImportError, OSError):
    pass

__all__ = [
    "CATEGORICAL_FEATURES",
    "FORBIDDEN_LEAKAGE_COLUMNS",
    "NUMERIC_FEATURES",
    "PlattCalibratedClassifier",
    "TargetingPolicyEngine",
    "build_preprocessor",
    "get_temporal_splits",
    "train_all_models",
    "validate_feature_columns",
    "validate_temporal_ordering",
]
