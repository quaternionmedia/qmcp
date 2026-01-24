"""API request/response schemas for the QMCP agent framework.

This submodule contains all Pydantic/SQLModel schemas organized by domain:
- agents: AgentType*, AgentInstance*, Skill*
- topologies: Topology*
- executions: Execution*, Result*, Checkpoint*
- messages: Message*
- workflows: WorkflowTemplate*
- metrics: Metric*, ToolInvocation*
- audit: AuditLogRead
- common: Pagination*, Filter*, Sort*, HealthCheck, SystemStats
"""

from .agents import (
    AgentInstanceRead,
    AgentInstanceSummary,
    AgentTypeCreate,
    AgentTypeRead,
    AgentTypeSummary,
    AgentTypeUpdate,
    SkillCreate,
    SkillRead,
)
from .audit import AuditLogRead
from .common import (
    FilterParams,
    HealthCheck,
    PaginatedResponse,
    PaginationParams,
    SortParams,
    SystemStats,
)
from .executions import (
    CheckpointRead,
    ExecutionCreate,
    ExecutionRead,
    ExecutionStatusUpdate,
    ExecutionSummary,
    ResultCreate,
    ResultRead,
)
from .messages import MessageCreate, MessageRead
from .metrics import (
    MetricAggregation,
    MetricCreate,
    MetricRead,
    ToolInvocationCreate,
    ToolInvocationRead,
)
from .topologies import (
    TopologyCreate,
    TopologyRead,
    TopologySummary,
    TopologyUpdate,
)
from .workflows import (
    WorkflowTemplateCreate,
    WorkflowTemplateRead,
    WorkflowTemplateUpdate,
)

__all__ = [
    # Agent Type
    "AgentTypeCreate",
    "AgentTypeUpdate",
    "AgentTypeRead",
    "AgentTypeSummary",
    # Agent Instance
    "AgentInstanceRead",
    "AgentInstanceSummary",
    # Skill
    "SkillCreate",
    "SkillRead",
    # Topology
    "TopologyCreate",
    "TopologyUpdate",
    "TopologyRead",
    "TopologySummary",
    # Execution
    "ExecutionCreate",
    "ExecutionRead",
    "ExecutionSummary",
    "ExecutionStatusUpdate",
    # Result
    "ResultCreate",
    "ResultRead",
    # Checkpoint
    "CheckpointRead",
    # Message
    "MessageCreate",
    "MessageRead",
    # Workflow Template
    "WorkflowTemplateCreate",
    "WorkflowTemplateUpdate",
    "WorkflowTemplateRead",
    # Metric
    "MetricCreate",
    "MetricRead",
    "MetricAggregation",
    # Tool Invocation
    "ToolInvocationCreate",
    "ToolInvocationRead",
    # Audit Log
    "AuditLogRead",
    # Pagination and Query
    "PaginationParams",
    "PaginatedResponse",
    "FilterParams",
    "SortParams",
    # Health and Status
    "HealthCheck",
    "SystemStats",
]
