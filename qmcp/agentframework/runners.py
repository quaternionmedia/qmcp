"""Runner skeletons for the QMCP agent framework."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeVar

from sqlmodel import Field, SQLModel

from qmcp.agentframework.models import AgentType, Topology
from qmcp.agentframework.topologies import TopologyRegistry


class RunConfig(SQLModel):
    """Configuration for runner execution."""

    timeout_seconds: int = 300
    max_retries: int = 3
    parallelism: int = 1
    save_intermediate: bool = True
    metadata_: dict[str, Any] = Field(default_factory=dict)


class RunResult(SQLModel):
    """Result from a runner execution."""

    execution_id: str | None = None
    status: str = "unknown"
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None


class BaseRunner(ABC):
    """Abstract base class for runners."""

    runner_type: ClassVar[str]

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig()

    async def run(self, topology: Topology, agents: dict[str, AgentType]) -> RunResult:
        if TopologyRegistry.get(topology.topology_type) is None:
            raise ValueError(f"No topology registered for {topology.topology_type}")
        raise NotImplementedError("Runner execution is not implemented yet.")


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


@runner
class LocalRunner(BaseRunner):
    runner_type = "local"


@runner
class AsyncRunner(BaseRunner):
    runner_type = "async"


@runner
class MetaflowRunner(BaseRunner):
    runner_type = "metaflow"


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
