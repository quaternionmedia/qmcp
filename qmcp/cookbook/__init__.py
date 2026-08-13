"""Composable building blocks for QMCP Metaflow flows.

Provides reusable modules for agent creation, flow persistence,
and MCP server integration that can be composed into workflows.

Usage:
    from qmcp.cookbook import (
        LocalLLMConfig, build_local_agent,
        FlowPersistence,
        MCPToolInvoker,
    )
"""

from qmcp.cookbook.agent_builders import LocalLLMConfig, build_local_agent, build_qmcp_agent
from qmcp.cookbook.mcp_tools import MCPToolInvoker, check_health, invoke_tool
from qmcp.cookbook.persistence import FlowPersistence, init_db
from qmcp.cookbook.steps import AgentPipeline, AgentStep, StepResult

__all__ = [
    # Agent builders
    "LocalLLMConfig",
    "build_local_agent",
    "build_qmcp_agent",
    # Persistence
    "FlowPersistence",
    "init_db",
    # MCP tools
    "MCPToolInvoker",
    "check_health",
    "invoke_tool",
    # Steps & pipelines
    "AgentStep",
    "AgentPipeline",
    "StepResult",
]
