"""Shared infrastructure and lifecycle components for CreditTarget."""

from credittarget.common.artifact_store import ArtifactStore, sha256_dict, sha256_text
from credittarget.common.base_stage import BasePipelineStage, StageResult

try:
    from credittarget.common.delta_io import DeltaTableManager
    from credittarget.common.dq import DataQualityChecker
    from credittarget.common.spark_utils import (
        assign_bins_by_edges,
        compute_reference_bin_edges,
        deduplicate_latest,
        export_physical_plan,
        safe_broadcast_join,
    )
except (ImportError, ModuleNotFoundError):
    pass

try:
    from credittarget.common.mlflow_client import ModelRegistryClient
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [
    "ArtifactStore",
    "BasePipelineStage",
    "DataQualityChecker",
    "DeltaTableManager",
    "ModelRegistryClient",
    "StageResult",
    "assign_bins_by_edges",
    "compute_reference_bin_edges",
    "deduplicate_latest",
    "export_physical_plan",
    "safe_broadcast_join",
    "sha256_dict",
    "sha256_text",
]
