"""FastAPI MCP Server.

This server exposes MCP-compatible endpoints for:
- Tool discovery
- Tool invocation
- Invocation history (audit)
- Human-in-the-loop requests and responses

The server is intentionally "boring":
- No orchestration logic
- No multi-step flows
- No agent chaining
- No hidden state
"""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, status
from sqlmodel import select

from qmcp import __version__
from qmcp.config import get_settings
from qmcp.db import HumanRequest, HumanResponse, ToolInvocation, get_session, init_db
from qmcp.db.engine import close_db
from qmcp.db.models import HumanRequestStatus, InvocationStatus
from qmcp.logging import configure_logging, get_logger
from qmcp.metrics import metrics, record_hitl_request, record_tool_invocation
from qmcp.middleware import RequestTracingMiddleware
from qmcp.schemas.mcp import (
    HumanRequestCreate,
    HumanRequestListResponse,
    HumanRequestResponse,
    HumanResponseCreate,
    HumanResponseResult,
    InvocationListResponse,
    ToolInvokeRequest,
    ToolInvokeResponse,
    ToolListResponse,
)

# Import builtin tools to register them
from qmcp.tools import builtin as _builtin_tools  # noqa: F401
from qmcp.tools import tool_registry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()

    # Configure logging based on environment
    configure_logging(
        json_format=not settings.debug,  # JSON in production, console in dev
        level="DEBUG" if settings.debug else "INFO",
    )

    # Startup
    logger.info(
        "server_starting",
        version=__version__,
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
    )
    logger.info(
        "tools_registered",
        tools=[t.name for t in tool_registry.list_tools()],
    )
    logger.info("database_init", database_url=settings.database_url)
    await init_db()
    logger.info("database_ready")
    yield
    # Shutdown
    await close_db()
    logger.info("server_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="QMCP Server",
        description="A spec-aligned Model Context Protocol server",
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Add request tracing middleware
    app.add_middleware(RequestTracingMiddleware)

    # Health check
    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    # Metrics endpoint (Prometheus format)
    @app.get("/metrics", include_in_schema=False)
    async def get_metrics() -> str:
        """Return Prometheus-formatted metrics."""
        from starlette.responses import PlainTextResponse
        return PlainTextResponse(
            content=metrics.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Metrics endpoint (JSON format)
    @app.get("/metrics/json")
    async def get_metrics_json() -> dict:
        """Return metrics as JSON."""
        return metrics.get_stats()

    # MCP Tool Discovery
    @app.get("/v1/tools", response_model=ToolListResponse)
    async def list_tools() -> ToolListResponse:
        """List all available MCP tools.

        Returns tool definitions including name, description, and input schema.
        """
        return ToolListResponse(tools=tool_registry.list_definitions())

    # MCP Tool Invocation
    @app.post("/v1/tools/{tool_name}", response_model=ToolInvokeResponse)
    async def invoke_tool(
        tool_name: str,
        request: ToolInvokeRequest,
    ) -> ToolInvokeResponse:
        """Invoke an MCP tool by name.

        Args:
            tool_name: The name of the tool to invoke
            request: The tool invocation request with input parameters

        Returns:
            The tool execution result

        Raises:
            404: Tool not found
            500: Tool execution error
        """
        tool = tool_registry.get(tool_name)
        if tool is None:
            logger.warning("tool_not_found", tool_name=tool_name)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found",
            )

        # Create invocation record
        invocation = ToolInvocation(
            tool_name=tool_name,
            input_params=request.input,
            correlation_id=request.correlation_id,
        )

        logger.info(
            "tool_invocation_start",
            tool_name=tool_name,
            invocation_id=invocation.id,
            correlation_id=request.correlation_id,
        )

        start_time = time.perf_counter()
        try:
            result = tool.invoke(request.input)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Update invocation record
            invocation.result = result
            invocation.status = InvocationStatus.SUCCESS
            invocation.duration_ms = elapsed_ms
            invocation.completed_at = datetime.now(UTC)

            # Persist to database
            async with get_session() as session:
                session.add(invocation)

            logger.info(
                "tool_invocation_success",
                tool_name=tool_name,
                invocation_id=invocation.id,
                duration_ms=elapsed_ms,
            )

            # Record metrics
            record_tool_invocation(tool_name, "success", elapsed_ms / 1000)

            return ToolInvokeResponse(
                result=result,
                invocation_id=invocation.id,
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Update invocation record with error
            invocation.error = str(e)
            invocation.status = InvocationStatus.FAILED
            invocation.duration_ms = elapsed_ms
            invocation.completed_at = datetime.now(UTC)

            # Persist to database
            async with get_session() as session:
                session.add(invocation)

            logger.error(
                "tool_invocation_failed",
                tool_name=tool_name,
                invocation_id=invocation.id,
                error=str(e),
                duration_ms=elapsed_ms,
            )

            # Record metrics
            record_tool_invocation(tool_name, "error", elapsed_ms / 1000)

            return ToolInvokeResponse(
                result=None,
                error=f"Tool execution failed: {str(e)}",
                invocation_id=invocation.id,
            )

    # Invocation History
    @app.get("/v1/invocations", response_model=InvocationListResponse)
    async def list_invocations(
        tool_name: str | None = Query(default=None, description="Filter by tool name"),
        status_filter: InvocationStatus | None = Query(
            default=None, alias="status", description="Filter by status"
        ),
        limit: int = Query(default=50, ge=1, le=500, description="Max results"),
        offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    ) -> InvocationListResponse:
        """List tool invocation history.

        Supports filtering by tool name and status, with pagination.
        """
        async with get_session() as session:
            query = select(ToolInvocation).order_by(ToolInvocation.created_at.desc())

            if tool_name:
                query = query.where(ToolInvocation.tool_name == tool_name)
            if status_filter:
                query = query.where(ToolInvocation.status == status_filter)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            invocations = result.scalars().all()

            return InvocationListResponse(
                invocations=list(invocations),
                count=len(invocations),
            )

    # Get single invocation
    @app.get("/v1/invocations/{invocation_id}")
    async def get_invocation(invocation_id: str) -> ToolInvocation:
        """Get a specific invocation by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(ToolInvocation).where(ToolInvocation.id == invocation_id)
            )
            invocation = result.scalar_one_or_none()

            if invocation is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Invocation '{invocation_id}' not found",
                )

            return invocation

    # =========================================================================
    # Human-in-the-Loop Endpoints
    # =========================================================================

    @app.post("/v1/human/requests", response_model=HumanRequestResponse, status_code=201)
    async def create_human_request(request: HumanRequestCreate) -> HumanRequestResponse:
        """Create a new human approval/input request.

        The request will be persisted and can be polled for a response.
        Requests expire after timeout_seconds.
        """
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=request.timeout_seconds)

        human_request = HumanRequest(
            id=request.id,
            request_type=request.request_type,
            prompt=request.prompt,
            options=request.options,
            context=request.context,
            timeout_seconds=request.timeout_seconds,
            correlation_id=request.correlation_id,
            created_at=now,
            expires_at=expires_at,
        )

        async with get_session() as session:
            # Check if request ID already exists
            existing = await session.execute(
                select(HumanRequest).where(HumanRequest.id == request.id)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Request with ID '{request.id}' already exists",
                )
            session.add(human_request)

        # Record metrics
        record_hitl_request(request.request_type, "created")

        return HumanRequestResponse(
            id=human_request.id,
            request_type=human_request.request_type,
            prompt=human_request.prompt,
            status=human_request.status.value,
            created_at=human_request.created_at,
            expires_at=human_request.expires_at,
        )

    @app.get("/v1/human/requests", response_model=HumanRequestListResponse)
    async def list_human_requests(
        status_filter: HumanRequestStatus | None = Query(
            default=None, alias="status", description="Filter by status"
        ),
        request_type: str | None = Query(default=None, description="Filter by type"),
        limit: int = Query(default=50, ge=1, le=500, description="Max results"),
        offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    ) -> HumanRequestListResponse:
        """List human requests with optional filtering."""
        async with get_session() as session:
            query = select(HumanRequest).order_by(HumanRequest.created_at.desc())

            if status_filter:
                query = query.where(HumanRequest.status == status_filter)
            if request_type:
                query = query.where(HumanRequest.request_type == request_type)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            requests = result.scalars().all()

            return HumanRequestListResponse(
                requests=list(requests),
                count=len(requests),
            )

    @app.get("/v1/human/requests/{request_id}")
    async def get_human_request(request_id: str) -> dict:
        """Get a human request by ID, including any response.

        Use this endpoint to poll for a response.
        """
        async with get_session() as session:
            # Get the request
            result = await session.execute(
                select(HumanRequest).where(HumanRequest.id == request_id)
            )
            request = result.scalar_one_or_none()

            if request is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Request '{request_id}' not found",
                )

            # Check if expired
            now = datetime.now(UTC)
            if (
                request.status == HumanRequestStatus.PENDING
                and request.expires_at
            ):
                # SQLite stores naive datetimes, treat as UTC
                expires_at = request.expires_at.replace(tzinfo=UTC) if request.expires_at.tzinfo is None else request.expires_at
                if expires_at < now:
                    request.status = HumanRequestStatus.EXPIRED

            # Get any response
            response_result = await session.execute(
                select(HumanResponse).where(HumanResponse.request_id == request_id)
            )
            response = response_result.scalar_one_or_none()

            return {
                "request": request,
                "response": response,
            }

    @app.post("/v1/human/responses", response_model=HumanResponseResult, status_code=201)
    async def submit_human_response(response: HumanResponseCreate) -> HumanResponseResult:
        """Submit a human response to a pending request."""
        async with get_session() as session:
            # Get the request
            result = await session.execute(
                select(HumanRequest).where(HumanRequest.id == response.request_id)
            )
            request = result.scalar_one_or_none()

            if request is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Request '{response.request_id}' not found",
                )

            # Check if already responded
            if request.status == HumanRequestStatus.RESPONDED:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Request has already been responded to",
                )

            # Check if expired
            now = datetime.now(UTC)
            if request.expires_at:
                # SQLite stores naive datetimes, treat as UTC
                expires_at = request.expires_at.replace(tzinfo=UTC) if request.expires_at.tzinfo is None else request.expires_at
                if expires_at < now:
                    request.status = HumanRequestStatus.EXPIRED
                    raise HTTPException(
                        status_code=status.HTTP_410_GONE,
                        detail="Request has expired",
                    )

            # Validate response against options if provided
            if request.options and response.response not in request.options:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Response must be one of: {request.options}",
                )

            # Create the response
            human_response = HumanResponse(
                request_id=response.request_id,
                response=response.response,
                responded_by=response.responded_by,
                response_metadata=response.response_metadata,
                created_at=now,
            )
            session.add(human_response)

            # Update request status
            request.status = HumanRequestStatus.RESPONDED

            # Record metrics
            record_hitl_request(request.request_type, "responded")

            return HumanResponseResult(
                id=human_response.id,
                request_id=human_response.request_id,
                response=human_response.response,
                responded_by=human_response.responded_by,
                created_at=human_response.created_at,
            )

    return app


# Application instance for uvicorn
app = create_app()
