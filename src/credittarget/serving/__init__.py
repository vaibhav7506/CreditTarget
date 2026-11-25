"""Serving layer: distributed batch scoring and low-latency FastAPI inference service."""

from credittarget.serving.api import app, handler
from credittarget.serving.batch_score import BatchScoreStage

__all__ = ["BatchScoreStage", "app", "handler"]
