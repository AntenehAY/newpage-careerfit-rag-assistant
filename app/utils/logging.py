"""Structured logging with loguru for Career Intelligence Assistant."""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _loguru_logger

if TYPE_CHECKING:
    from loguru import Logger

# Context variables for request-scoped metadata (set by middleware)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_operation: ContextVar[str | None] = ContextVar("operation", default=None)

# Default log dir
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
ROTATION = "10 MB"
RETENTION = "7 days"


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | str | None = None,
) -> None:
    """Configure loguru with structured logs, file rotation, and console output.

    - JSON format for file logs (structured, parseable)
    - Pretty format for console (dev-friendly)
    - File rotation: 10 MB max per file
    - Retention: 7 days
    - Output: logs/app.log (or log_dir/app.log if log_dir provided)

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Optional directory for log files (used for tests).
    """
    # Remove default handler
    _loguru_logger.remove()

    log_path = Path(log_dir) if log_dir else LOG_DIR
    log_file = log_path / "app.log"
    log_path.mkdir(parents=True, exist_ok=True)

    def _enrich_record(record: dict) -> bool:
        """Add context vars to record before serialization."""
        rid = _request_id.get()
        uid = _user_id.get()
        op = _operation.get()
        if rid is not None:
            record["extra"]["request_id"] = rid
        if uid is not None:
            record["extra"]["user_id"] = uid
        if op is not None:
            record["extra"]["operation"] = op
        return True

    _loguru_logger.add(
        log_file,
        format="{message}",
        rotation=ROTATION,
        retention=RETENTION,
        level=log_level,
        serialize=True,
        filter=_enrich_record,
    )

    _loguru_logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[module]}</cyan> | {message}",
        level=log_level,
    )


def get_logger(module_name: str) -> Logger:
    """Return a configured logger for the given module.

    Adds context: module name, timestamp. Callers can bind request_id,
    user_id, operation via context vars (set by middleware) or manually
    via logger.bind().

    Args:
        module_name: Typically __name__ of the calling module.

    Returns:
        Logger instance with module context.
    """
    return _loguru_logger.bind(module=module_name)


def set_request_context(
    request_id: str | None = None,
    user_id: str | None = None,
    operation: str | None = None,
) -> None:
    """Set context for the current request/operation (used by middleware)."""
    if request_id is not None:
        _request_id.set(request_id)
    if user_id is not None:
        _user_id.set(user_id)
    if operation is not None:
        _operation.set(operation)


def clear_request_context() -> None:
    """Clear request context (call at end of request)."""
    try:
        _request_id.set(None)
    except LookupError:
        pass
    try:
        _user_id.set(None)
    except LookupError:
        pass
    try:
        _operation.set(None)
    except LookupError:
        pass
