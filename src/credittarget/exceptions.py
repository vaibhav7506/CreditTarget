"""Typed domain exceptions for CreditTarget."""


class CreditTargetError(Exception):
    """Base exception for all CreditTarget errors."""


class ConfigError(CreditTargetError):
    """Raised when configuration is invalid or missing."""


class SchemaContractError(CreditTargetError):
    """Raised when data violates explicit contract schemas."""


class DataQualityError(CreditTargetError):
    """Raised when data quality assertions fail."""


class LeakageDetectedError(CreditTargetError):
    """Raised when target leakage or future-timestamp leakage is detected."""


class GovernanceGateError(CreditTargetError):
    """Raised when model risk governance thresholds are violated."""


class ReconciliationError(CreditTargetError):
    """Raised when GenAI validation memo reports numbers inconsistent with artifacts."""


class ModelNotFoundError(CreditTargetError):
    """Raised when requested model or alias cannot be loaded from registry."""
