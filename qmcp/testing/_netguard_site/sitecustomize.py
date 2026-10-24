"""Guards a Python child of the suite: `qmcp.testing.netguard`, installed first.

Loaded by file path rather than imported as `qmcp.testing.netguard`, so a child
starts without importing the `qmcp` package, and a child whose interpreter
cannot import it is guarded all the same.
"""

import importlib.util
import os
import pathlib

_path = pathlib.Path(__file__).resolve().parent.parent / "netguard.py"
_spec = importlib.util.spec_from_file_location("_qmcp_netguard", _path)
_netguard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_netguard)
_ports = tuple(int(port) for port in os.environ.get(_netguard.PORTS, "").split(",") if port)
_netguard.install(os.environ.get(_netguard.LOG), _ports)
