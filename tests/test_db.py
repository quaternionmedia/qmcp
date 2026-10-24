"""Tests for database persistence."""

from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from qmcp.db.models import (
    HumanRequest,
    HumanRequestStatus,
    HumanResponse,
    InvocationStatus,
    ToolInvocation,
)

# Test session factory - set by fixture
_test_session_factory = None


@asynccontextmanager
async def get_test_session():
    """Get a test database session."""
    async with _test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Initialize a fresh in-memory database for each test."""
    global _test_session_factory

    # Create a fresh in-memory engine for each test
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    # Create session factory
    _test_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield

    await engine.dispose()
    _test_session_factory = None


@pytest.mark.asyncio
class TestToolInvocation:
    """Tests for ToolInvocation model."""

    async def test_create_invocation(self):
        """Test creating a tool invocation record."""
        async with get_test_session() as session:
            invocation = ToolInvocation(
                tool_name="echo",
                input_params={"message": "hello"},
            )
            session.add(invocation)

        # Verify it was saved
        async with get_test_session() as session:
            result = await session.execute(select(ToolInvocation))
            invocations = result.scalars().all()
            assert len(invocations) == 1
            assert invocations[0].tool_name == "echo"
            assert invocations[0].input_params == {"message": "hello"}

    async def test_invocation_status_transitions(self):
        """Test invocation status updates."""
        async with get_test_session() as session:
            invocation = ToolInvocation(
                tool_name="test",
                input_params={},
                status=InvocationStatus.PENDING,
            )
            session.add(invocation)
            inv_id = invocation.id

        async with get_test_session() as session:
            result = await session.execute(
                select(ToolInvocation).where(ToolInvocation.id == inv_id)
            )
            invocation = result.scalar_one()
            invocation.status = InvocationStatus.SUCCESS
            invocation.result = {"success": True}

        async with get_test_session() as session:
            result = await session.execute(
                select(ToolInvocation).where(ToolInvocation.id == inv_id)
            )
            invocation = result.scalar_one()
            assert invocation.status == InvocationStatus.SUCCESS

    async def test_query_by_tool_name(self):
        """Test querying invocations by tool name."""
        async with get_test_session() as session:
            session.add(ToolInvocation(tool_name="echo", input_params={}))
            session.add(ToolInvocation(tool_name="planner", input_params={}))
            session.add(ToolInvocation(tool_name="echo", input_params={}))

        async with get_test_session() as session:
            result = await session.execute(
                select(ToolInvocation).where(ToolInvocation.tool_name == "echo")
            )
            echo_invocations = result.scalars().all()
            assert len(echo_invocations) == 2


@pytest.mark.asyncio
class TestHumanRequest:
    """Tests for HumanRequest model."""

    async def test_create_human_request(self):
        """Test creating a human request."""
        async with get_test_session() as session:
            request = HumanRequest(
                id="approve-deploy-123",
                request_type="approval",
                prompt="Approve deployment to production?",
                options=["approve", "reject"],
                timeout_seconds=3600,
            )
            session.add(request)

        async with get_test_session() as session:
            result = await session.execute(
                select(HumanRequest).where(HumanRequest.id == "approve-deploy-123")
            )
            req = result.scalar_one()
            assert req.prompt == "Approve deployment to production?"
            assert req.options == ["approve", "reject"]
            assert req.status == HumanRequestStatus.PENDING

    async def test_human_request_with_context(self):
        """Test human request with additional context."""
        async with get_test_session() as session:
            request = HumanRequest(
                id="review-doc-456",
                request_type="review",
                prompt="Please review this document",
                context={"doc_id": "doc-123", "author": "user@example.com"},
            )
            session.add(request)

        async with get_test_session() as session:
            result = await session.execute(
                select(HumanRequest).where(HumanRequest.id == "review-doc-456")
            )
            req = result.scalar_one()
            assert req.context["doc_id"] == "doc-123"


@pytest.mark.asyncio
class TestHumanResponse:
    """Tests for HumanResponse model."""

    async def test_create_response_for_request(self):
        """Test creating a response linked to a request."""
        # Create request first
        async with get_test_session() as session:
            request = HumanRequest(
                id="req-001",
                request_type="approval",
                prompt="Approve?",
            )
            session.add(request)

        # Create response
        async with get_test_session() as session:
            response = HumanResponse(
                request_id="req-001",
                response="approve",
                responded_by="alice@example.com",
            )
            session.add(response)

        # Verify
        async with get_test_session() as session:
            result = await session.execute(
                select(HumanResponse).where(HumanResponse.request_id == "req-001")
            )
            resp = result.scalar_one()
            assert resp.response == "approve"
            assert resp.responded_by == "alice@example.com"


def test_the_server_creates_its_own_tables_and_not_the_cookbooks(tmp_path, monkeypatch):
    """`SQLModel.metadata` holds the cookbook's flow tables once anything has
    imported the cookbook. Found by tests/test_integration_migrated_database.py,
    which failed only in the full suite: a server started after the cookbook
    was imported wrote `flowrun`, `agentrun`, `mcpinvocation` and `artifact`
    into its database. Mutation: `init_db` back on `SQLModel.metadata` -- red."""
    import asyncio
    import sqlite3

    import qmcp.cookbook.persistence  # noqa: F401 -- registering the flow tables is the point
    import qmcp.db.engine as engine_module
    from qmcp.config import Settings
    from qmcp.db.registry import server_table_names

    database = tmp_path / "server.db"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{database.as_posix()}")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings)
    monkeypatch.setattr(engine_module, "_engine", None)

    async def start_and_stop() -> None:
        await engine_module.init_db()
        await engine_module.close_db()

    asyncio.run(start_and_stop())
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert tables & {"flowrun", "agentrun", "mcpinvocation", "artifact", "checklistitem"} == set()
    assert server_table_names() <= tables
