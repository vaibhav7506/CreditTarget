"""Serving layer: distributed batch scoring and low-latency FastAPI inference service."""

from credittarget.serving.api import app, handler

try:
    from credittarget.serving.batch_score import BatchScoreStage
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ["BatchScoreStage", "app", "handler"]
