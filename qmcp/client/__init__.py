"""QMCP Client Library.

Provides an HTTP client for interacting with the MCP server.

The errors are exported as well as the client, so a caller can name the
failure it handles: `examples/flows/approved_deploy.py` imports
`HumanRequestExpiredError` from this package.
"""

from qmcp.client.mcp_client import (
    HumanRequestConflictError,
    HumanRequestExpiredError,
    MCPClient,
    MCPClientError,
    ToolNotFoundError,
)

__all__ = [
    "MCPClient",
    "MCPClientError",
    "ToolNotFoundError",
    "HumanRequestExpiredError",
    "HumanRequestConflictError",
]
