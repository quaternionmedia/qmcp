"""Tool registration system.

Tools are:
- Stateless
- Pure functions where possible
- Do not orchestrate or call other tools
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qmcp.schemas.mcp import ToolDefinition


@dataclass
class Tool:
    """A registered MCP tool."""

    name: str
    description: str
    handler: Callable[[dict[str, Any]], Any]
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_definition(self) -> ToolDefinition:
        """Convert to MCP tool definition for discovery."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def invoke(self, params: dict[str, Any]) -> Any:
        """Invoke the tool with given parameters."""
        return self.handler(params)


class ToolRegistry:
    """Registry for MCP tools.

    Provides:
    - Tool registration via decorator
    - Tool discovery
    - Tool invocation
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any] | None = None,
    ) -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
        """Decorator to register a tool.

        Usage:
            @registry.register("echo", "Echo the input back")
            def echo(params: dict) -> str:
                return params.get("message", "")
        """

        def decorator(
            func: Callable[[dict[str, Any]], Any],
        ) -> Callable[[dict[str, Any]], Any]:
            tool = Tool(
                name=name,
                description=description,
                handler=func,
                input_schema=input_schema or {},
            )
            self._tools[name] = tool
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_definitions(self) -> list[ToolDefinition]:
        """List all tool definitions for discovery."""
        return [tool.to_definition() for tool in self._tools.values()]


# Global registry instance
tool_registry = ToolRegistry()
