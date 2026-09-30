"""QMCP - Model Context Protocol Server.

This module provides backward compatibility.
Use `uv run qmcp <command>` instead; `uv run qmcp --help` lists them.
"""

from qmcp.cli import main

if __name__ == "__main__":
    main()
