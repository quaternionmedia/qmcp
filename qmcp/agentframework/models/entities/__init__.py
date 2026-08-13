"""Database entity models for the QMCP agent framework.

This submodule contains all SQLModel table definitions organized by domain:
- agents: AgentType, AgentInstance, AgentSkill
- topologies: Topology, TopologyMembership
- executions: Execution, Result, Checkpoint
- messages: Message
- resources: ResourceAllocation
- audit: AuditLog, MetricRecord, AgentToolInvocation
- workflows: WorkflowTemplate
"""

from .agents import AgentInstance, AgentSkill, AgentType
from .audit import AuditLog, MetricRecord, AgentToolInvocation
from .executions import Checkpoint, Execution, Result
from .messages import Message
from .resources import ResourceAllocation
from .topologies import Topology, TopologyMembership
from .workflows import WorkflowTemplate

__all__ = [
    # Agents
    "AgentType",
    "AgentInstance",
    "AgentSkill",
    # Topologies
    "Topology",
    "TopologyMembership",
    # Executions
    "Execution",
    "Result",
    "Checkpoint",
    # Messages
    "Message",
    # Resources
    "ResourceAllocation",
    # Audit
    "AuditLog",
    "MetricRecord",
    "AgentToolInvocation",
    # Workflows
    "WorkflowTemplate",
]
