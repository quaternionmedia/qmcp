"""Runner implementations for the QMCP agent framework.

Runners execute topologies by launching the appropriate Metaflow flow
or running agent logic in-process.  The MetaflowRunner is the primary
production runner — it resolves a topology to a flow file and executes
it via subprocess (locally or inside Docker), mirroring the CLI's
cookbook/council commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, TypeVar
from uuid import uuid4

from sqlmodel import Field, SQLModel

from qmcp.agentframework.models import AgentType, Topology, TopologyType
from qmcp.agentframework.topologies import TopologyRegistry


# ---------------------------------------------------------------------------
# Configuration & result models
# ---------------------------------------------------------------------------


class RunConfig(SQLModel):
    """Configuration for runner execution."""

    timeout_seconds: int = 300
    max_retries: int = 3
    parallelism: int = 1
    save_intermediate: bool = True
    metadata_: dict[str, Any] = Field(default_factory=dict)
    # Metaflow-specific
    docker: bool = False
    mcp_url: str | None = None
    metaflow_user: str | None = None
    build_image: bool = True
    sync_deps: bool = True


class RunResult(SQLModel):
    """Result from a runner execution."""

    execution_id: str | None = None
    status: str = "unknown"
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseRunner(ABC):
    """Abstract base class for runners."""

    runner_type: ClassVar[str]

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()

    async def run(self, topology: Topology, agents: dict[str, AgentType]) -> RunResult:
        if TopologyRegistry.get(topology.topology_type) is None:
            raise ValueError(f"No topology registered for {topology.topology_type}")
        raise NotImplementedError("Runner execution is not implemented yet.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

R = TypeVar("R", bound=BaseRunner)


class RunnerRegistry:
    """Registry for runner implementations."""

    _runners: dict[str, type[BaseRunner]] = {}

    @classmethod
    def register(cls, runner_class: type[BaseRunner]) -> type[BaseRunner]:
        cls._runners[runner_class.runner_type] = runner_class
        return runner_class

    @classmethod
    def get(cls, runner_type: str) -> type[BaseRunner] | None:
        return cls._runners.get(runner_type)

    @classmethod
    def create(cls, runner_type: str, config: RunConfig | None = None) -> BaseRunner:
        runner_class = cls.get(runner_type)
        if runner_class is None:
            raise ValueError(f"Unknown runner type: {runner_type}")
        return runner_class(config=config)


def runner(cls: type[R]) -> type[R]:
    """Decorator to register a runner implementation."""
    return RunnerRegistry.register(cls)


# ---------------------------------------------------------------------------
# Default topology → flow mapping
# ---------------------------------------------------------------------------

_TOPOLOGY_FLOW_MAP: dict[TopologyType, str] = {
    TopologyType.PIPELINE: "examples/flows/local_agent_chain.py",
    TopologyType.COUNCIL: "examples/flows/council_deliberation.py",
}


# ---------------------------------------------------------------------------
# Runner implementations
# ---------------------------------------------------------------------------


@runner
class LocalRunner(BaseRunner):
    """Execute a topology in-process (skeleton — topology runtime pending)."""

    runner_type = "local"


@runner
class AsyncRunner(BaseRunner):
    """Execute a topology asynchronously (skeleton — topology runtime pending)."""

    runner_type = "async"


@runner
class MetaflowRunner(BaseRunner):
    """Execute a topology as a Metaflow flow via subprocess.

    Resolves the topology type to a flow script and runs it either
    locally (``python <flow> run``) or inside Docker, using the same
    mechanism as the CLI's ``qmcp cookbook run`` / ``qmcp council run``.

    Args:
        config: Runner configuration (includes Metaflow-specific fields).
        flow_path: Explicit path to the Metaflow flow script.  If *None*,
            the runner resolves the path from ``_TOPOLOGY_FLOW_MAP``.
    """

    runner_type = "metaflow"

    def __init__(
        self,
        config: RunConfig | None = None,
        flow_path: str | None = None,
    ) -> None:
        super().__init__(config)
        self.flow_path = flow_path

    # -- public API --------------------------------------------------------

    async def run(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        flow_args: list[str] | None = None,
    ) -> RunResult:
        """Launch the Metaflow flow for *topology*.

        Args:
            topology: The topology configuration to execute.
            agents: Agent type definitions (used for metadata only today).
            flow_args: Extra CLI arguments forwarded to the flow script.

        Returns:
            A ``RunResult`` with execution status and timing.
        """
        execution_id = str(uuid4())
        start = time.monotonic()

        resolved_path = self._resolve_flow_path(topology)
        cmd = self._build_command(resolved_path, flow_args or [])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            if result.returncode == 0:
                return RunResult(
                    execution_id=execution_id,
                    status="completed",
                    output={
                        "stdout": result.stdout,
                        "returncode": result.returncode,
                    },
                    duration_ms=duration_ms,
                )
            else:
                return RunResult(
                    execution_id=execution_id,
                    status="failed",
                    output={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    },
                    error=result.stderr or f"Exit code {result.returncode}",
                    duration_ms=duration_ms,
                )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return RunResult(
                execution_id=execution_id,
                status="failed",
                error=f"Flow timed out after {self.config.timeout_seconds}s",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return RunResult(
                execution_id=execution_id,
                status="failed",
                error=str(exc),
                duration_ms=duration_ms,
            )

    # -- internals ---------------------------------------------------------

    def _resolve_flow_path(self, topology: Topology) -> str:
        """Return the flow script path for *topology*."""
        if self.flow_path:
            return self.flow_path

        default = _TOPOLOGY_FLOW_MAP.get(topology.topology_type)
        if default is None:
            raise ValueError(
                f"No default flow registered for topology "
                f"{topology.topology_type.value!r}. Pass flow_path explicitly."
            )
        return default

    def _build_command(
        self,
        flow_path: str,
        flow_args: list[str],
    ) -> list[str]:
        """Assemble the subprocess command list."""
        # Derived rather than typed in: a literal here could not reach the
        # server this package starts. `MCP_URL` still wins when it is set.
        from qmcp.client.mcp_client import default_base_url

        mcp_url = self.config.mcp_url or os.getenv("MCP_URL", default_base_url())

        cmd = [sys.executable, flow_path, "run"]
        cmd.extend(flow_args)

        # Inject --mcp-url if not already provided
        if "--mcp-url" not in flow_args:
            cmd.extend(["--mcp-url", mcp_url])

        # Set METAFLOW_USER in environment via config
        env_user = (
            self.config.metaflow_user
            or os.getenv("METAFLOW_USER")
            or os.getenv("USERNAME")
            or os.getenv("USER")
            or "local"
        )
        os.environ.setdefault("METAFLOW_USER", env_user)

        return cmd


__all__ = [
    "RunConfig",
    "RunResult",
    "BaseRunner",
    "RunnerRegistry",
    "runner",
    "LocalRunner",
    "AsyncRunner",
    "MetaflowRunner",
]
