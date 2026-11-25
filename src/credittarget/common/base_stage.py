"""Base pipeline stage contract with unified execution lifecycle, timing, and logging."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from credittarget.config import AppConfig
from credittarget.logging_utils import clear_run_context, get_logger, set_run_context


class StageResult(BaseModel):
    stage_name: str
    run_id: str
    snapshot_month: str | None = None
    status: str = "SUCCESS"  # SUCCESS or FAILED
    input_rows: int = 0
    output_rows: int = 0
    quarantine_rows: int = 0
    duration_sec: float = 0.0
    delta_versions: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class BasePipelineStage(ABC):
    """Abstract base class for all pipeline execution stages."""

    def __init__(self, config: AppConfig, stage_name: str | None = None) -> None:
        self.config = config
        self.stage_name = stage_name or self.__class__.__name__
        self.logger = get_logger(self.stage_name)

    @abstractmethod
    def extract(self, *args: Any, **kwargs: Any) -> Any:
        """Extract input data from sources or prior Delta tables."""

    @abstractmethod
    def transform(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply business logic, transformations, and joins."""

    @abstractmethod
    def validate(self, data: Any, *args: Any, **kwargs: Any) -> None:
        """Run data quality assertions and schema validation."""

    @abstractmethod
    def load(self, data: Any, *args: Any, **kwargs: Any) -> StageResult:
        """Persist output to Delta tables or artifacts and return StageResult."""

    def run(self, *args: Any, **kwargs: Any) -> StageResult:
        """Lifecycle runner: wraps extract -> transform -> validate -> load."""
        run_id = kwargs.get("run_id") or str(uuid.uuid4())
        snapshot_month = kwargs.get("snapshot_month")
        set_run_context(run_id=run_id, stage_name=self.stage_name)

        start_time = time.perf_counter()
        self.logger.info(
            "stage_started",
            stage_name=self.stage_name,
            snapshot_month=snapshot_month,
        )

        try:
            raw_data = self.extract(*args, **kwargs)
            transformed_data = self.transform(raw_data, *args, **kwargs)
            self.validate(transformed_data, *args, **kwargs)
            result = self.load(transformed_data, *args, **kwargs)

            duration = round(time.perf_counter() - start_time, 3)
            result.duration_sec = duration
            result.run_id = run_id
            if snapshot_month and not result.snapshot_month:
                result.snapshot_month = snapshot_month

            self.logger.info(
                "stage_completed",
                stage_name=self.stage_name,
                status=result.status,
                output_rows=result.output_rows,
                quarantine_rows=result.quarantine_rows,
                duration_sec=duration,
            )
            return result
        except Exception as e:
            duration = round(time.perf_counter() - start_time, 3)
            self.logger.exception(
                "stage_failed",
                stage_name=self.stage_name,
                duration_sec=duration,
                error=str(e),
            )
            return StageResult(
                stage_name=self.stage_name,
                run_id=run_id,
                snapshot_month=snapshot_month,
                status="FAILED",
                duration_sec=duration,
                error_message=str(e),
            )
        finally:
            clear_run_context()
