"""Local SQLModel storage for dev-cycle Metaflow flows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(UTC)


def new_id() -> str:
    """Generate a UUID string."""
    return str(uuid4())


class FlowRun(SQLModel, table=True):
    """Tracks a single Metaflow run."""

    id: str = Field(default_factory=new_id, primary_key=True)
    flow_name: str = Field(index=True)
    run_id: str = Field(index=True)
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class AgentRun(SQLModel, table=True):
    """Captures one agent invocation and its output."""

    id: str = Field(default_factory=new_id, primary_key=True)
    flow_run_id: str = Field(index=True)
    agent_name: str = Field(index=True)
    input_summary: str
    output: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class Artifact(SQLModel, table=True):
    """Materialized output for later inspection."""

    id: str = Field(default_factory=new_id, primary_key=True)
    flow_run_id: str = Field(index=True)
    kind: str = Field(index=True)
    content: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class ChecklistItem(SQLModel, table=True):
    """QC checklist row with execution guidance."""

    id: str = Field(default_factory=new_id, primary_key=True)
    flow_run_id: str = Field(index=True)
    area: str = Field(index=True)
    check: str
    command: str | None = None
    expected: str | None = None
    status: str = Field(default="pending", index=True)
    notes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MCPInvocation(SQLModel, table=True):
    """Tracks MCP tool invocations tied to a flow run."""

    id: str = Field(default_factory=new_id, primary_key=True)
    flow_run_id: str = Field(index=True)
    tool_name: str = Field(index=True)
    invocation_id: str = Field(index=True)
    correlation_id: str | None = Field(default=None, index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


def get_engine(db_path: str):
    """Create a SQLModel engine for the local flow database."""
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str):
    """Initialize tables and return the engine."""
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
    return engine


def save_flow_run(
    session: Session,
    flow_name: str,
    run_id: str,
    meta: dict[str, Any],
) -> FlowRun:
    """Insert a flow run row."""
    flow_run = FlowRun(flow_name=flow_name, run_id=run_id, meta=meta)
    session.add(flow_run)
    session.commit()
    session.refresh(flow_run)
    return flow_run


def mark_flow_finished(session: Session, flow_run_id: str) -> None:
    """Set finished_at for a flow run."""
    flow_run = session.get(FlowRun, flow_run_id)
    if flow_run is None:
        return
    flow_run.finished_at = utc_now()
    session.add(flow_run)
    session.commit()


def save_agent_run(
    session: Session,
    flow_run_id: str,
    agent_name: str,
    input_summary: str,
    output: dict[str, Any],
) -> AgentRun:
    """Insert an agent run record."""
    run = AgentRun(
        flow_run_id=flow_run_id,
        agent_name=agent_name,
        input_summary=input_summary,
        output=output,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def save_artifact(
    session: Session,
    flow_run_id: str,
    kind: str,
    content: dict[str, Any],
) -> Artifact:
    """Insert a materialized artifact row."""
    artifact = Artifact(flow_run_id=flow_run_id, kind=kind, content=content)
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def save_checklist_items(
    session: Session,
    flow_run_id: str,
    items: list[dict[str, Any]],
) -> None:
    """Insert QC checklist rows."""
    for item in items:
        record = ChecklistItem(flow_run_id=flow_run_id, **item)
        session.add(record)
    session.commit()


def save_mcp_invocation(
    session: Session,
    flow_run_id: str,
    tool_name: str,
    invocation_id: str,
    payload: dict[str, Any],
    correlation_id: str | None = None,
) -> MCPInvocation:
    """Insert a record for an MCP tool invocation."""
    record = MCPInvocation(
        flow_run_id=flow_run_id,
        tool_name=tool_name,
        invocation_id=invocation_id,
        payload=payload,
        correlation_id=correlation_id,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record
