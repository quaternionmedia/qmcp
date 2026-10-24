"""A qmcp server in a process of its own: `qmcp serve --port 0`, found by its run-file.

`throwaway_server` runs uvicorn in a thread of the caller, which shares the
caller's settings, engine and monkeypatches -- it cannot show what survives a
restart, or what a second program sees. This starts the real command in a
process of its own over the database it is given, waits for the run-file it
leaves (`qmcp/instance.py`), checks that `/health` on the port the run-file
names states the same identity, and stops it the way a person does: Ctrl-Break
on Windows, SIGINT elsewhere, so it shuts down cleanly and takes its run-file
away.

What this cannot do: start a server that already has the caller's
monkeypatches. Whatever the process should see goes in through the
environment.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_IDENTITY_KEYS = ("pid", "port", "started_at", "database_fingerprint", "session")


@dataclass
class Served:
    url: str
    run_file: Path
    identity: dict[str, Any]
    process: subprocess.Popen


def _stop(process: subprocess.Popen, timeout: float) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


@contextlib.contextmanager
def server_process(database: Path, run_dir: Path, session: str,
                   timeout: float = 30.0, env: dict[str, str] | None = None) -> Iterator[Served]:
    """Start `qmcp serve --port 0` over `database`; yield it once it says who it is."""
    environment = {**os.environ, **(env or {}),
                   "QMCP_DATABASE_URL": f"sqlite+aiosqlite:///{database.as_posix()}",
                   "QMCP_RUN_DIR": str(run_dir), "QMCP_SESSION": session}
    log = database.with_suffix(".server.log").open("w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, "-m", "qmcp", "serve", "--port", "0"],
                               env=environment, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=flags)
    try:
        deadline = time.monotonic() + timeout
        found: dict[str, Any] | None = None
        while found is None:
            if process.poll() is not None:
                raise RuntimeError(f"qmcp serve exited {process.returncode}; see {log.name}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"qmcp serve left no run-file in {timeout:.0f} s")
            for candidate in run_dir.glob("qmcp-*.json"):
                with contextlib.suppress(OSError, ValueError):
                    recorded = json.loads(candidate.read_text(encoding="utf-8"))
                    if recorded.get("session") == session:
                        found, run_file = recorded, candidate
            time.sleep(0.1)
        url = f"http://127.0.0.1:{found['port']}"
        stated = httpx.get(f"{url}/health", timeout=5).json()
        mismatched = [key for key in _IDENTITY_KEYS if stated.get(key) != found.get(key)]
        if mismatched:
            raise RuntimeError(f"{url} is not the server the run-file names: {mismatched}")
        yield Served(url=url, run_file=run_file, identity=found, process=process)
    finally:
        _stop(process, timeout)
        log.close()
