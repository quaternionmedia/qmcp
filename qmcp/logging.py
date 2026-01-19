"""Structured logging configuration for QMCP.

Uses structlog for JSON-formatted logs in production and
human-readable logs in development.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    json_format: bool = True,
    level: str = "INFO",
    add_timestamp: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        json_format: If True, output JSON logs; otherwise use console format.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        add_timestamp: Whether to add timestamps to log entries.
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Build processor chain
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        # JSON format for production
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Console format for development
        shared_processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.

    Useful for adding request_id, correlation_id, user_id, etc.

    Args:
        **kwargs: Context variables to bind.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """Remove specific context variables.

    Args:
        *keys: Context variable names to remove.
    """
    structlog.contextvars.unbind_contextvars(*keys)
