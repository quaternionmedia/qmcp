"""Tool system for QMCP."""

from qmcp.tools.registry import Tool, ToolRegistry, tool_registry


def published_registry() -> ToolRegistry:
    """The registry holding every tool the server publishes.

    Importing a tool module registers its tools, so this imports each one the
    server serves -- the built-in tools, the tools that drive alfred and the
    tools that read apothecary -- and returns the shared registry. The server
    and `qmcp tools list` both read it, so the two list the same tools.
    """
    from qmcp.tools import alfred as _alfred  # noqa: F401
    from qmcp.tools import apothecary as _apothecary  # noqa: F401
    from qmcp.tools import builtin as _builtin  # noqa: F401

    return tool_registry


__all__ = ["Tool", "ToolRegistry", "published_registry", "tool_registry"]
