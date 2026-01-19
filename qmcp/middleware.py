"""Request middleware for tracing and logging."""

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from qmcp.logging import bind_context, clear_context, get_logger
from qmcp.metrics import record_request

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request tracing and structured logging.

    Features:
    - Generates unique request_id for each request
    - Extracts correlation_id from X-Correlation-ID header
    - Logs request start and completion with timing
    - Adds request context to all logs during request handling
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with tracing context."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]

        # Extract correlation ID from header or generate one
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:8]

        # Bind context for all logs during this request
        clear_context()
        bind_context(
            request_id=request_id,
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
        )

        # Store in request state for handlers to access
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        # Log request start
        logger.info(
            "request_started",
            client_host=request.client.host if request.client else None,
            query_params=dict(request.query_params) if request.query_params else None,
        )

        # Time the request
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Record metrics (skip /metrics endpoint to avoid self-referential noise)
            if not request.url.path.startswith("/metrics"):
                record_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration=duration_ms / 1000,  # Convert to seconds
                )

            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                "request_failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
            )
            raise

        finally:
            # Clear context after request
            clear_context()


def get_request_id(request: Request) -> str | None:
    """Get the request ID from the current request.

    Args:
        request: The Starlette request object.

    Returns:
        The request ID or None if not set.
    """
    return getattr(request.state, "request_id", None)


def get_correlation_id(request: Request) -> str | None:
    """Get the correlation ID from the current request.

    Args:
        request: The Starlette request object.

    Returns:
        The correlation ID or None if not set.
    """
    return getattr(request.state, "correlation_id", None)
