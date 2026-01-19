"""QMCP - Model Context Protocol Server.

This module provides backward compatibility.
Use `qmcp serve` or `python -m qmcp.cli` instead.
"""

from qmcp.cli import main

if __name__ == "__main__":
    main()
