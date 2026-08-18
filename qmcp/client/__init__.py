"""QMCP Client Library.

Provides an HTTP client for interacting with the MCP server.

THE ERRORS ARE EXPORTED, NOT ONLY THE CLIENT. A caller that handles a failure
has to be able to name it, and `examples/flows/approved_deploy.py` imports
`HumanRequestExpiredError` from this package to do exactly that. Exporting only
`MCPClient` left that import broken -- and invisible on any machine where the
flow cannot be imported at all, which on Windows is every machine, because
`import metaflow` fails on `import fcntl`. The first CI run on Linux found it.
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
