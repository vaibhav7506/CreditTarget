"""Structured logging setup with correlation context for CreditTarget."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configures structlog to output structured JSON with timestamps and run_id."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retrieve a bound structured logger."""
    return structlog.get_logger(name)


def set_run_context(run_id: str, stage_name: str | None = None) -> None:
    """Binds correlation context variables to all logging downstream."""
    ctx: dict[str, Any] = {"run_id": run_id}
    if stage_name:
        ctx["stage_name"] = stage_name
    structlog.contextvars.bind_contextvars(**ctx)


def clear_run_context() -> None:
    """Clears all bound context variables."""
    structlog.contextvars.clear_contextvars()
