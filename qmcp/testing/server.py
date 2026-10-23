"""A qmcp server for one check: its own port, its own database, in this process.

Used by the product's offline checks (`qmcp cookbook voice`, `qmcp cookbook
instruct`), by the walkthrough pages that need a server, and by the suite's
integration tests -- every caller that must not write into the configured
queue or bind the configured port.

What this cannot do: stand for a separate server process. The server runs in
a thread of the caller's process, sees the caller's monkeypatches, and shares
its settings cache and database engine, which are swapped for the run and put
back. A path that crosses a process boundary needs a server started as one.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def throwaway_server(database: Path) -> Iterator[str]:
    """A qmcp server on an ephemeral port over its own database. Yields its URL.

    The configured database is somebody's queue, and a check must not write
    into it. The settings cache and the database engine are process-wide, so
    both are swapped for the run and put back afterwards.
    """
    import uvicorn

    import qmcp.config
    import qmcp.db.engine as db_engine
    from qmcp.server import create_app

    saved_url = os.environ.get("QMCP_DATABASE_URL")
    saved_engine = db_engine._engine
    os.environ["QMCP_DATABASE_URL"] = f"sqlite+aiosqlite:///{database.as_posix()}"
    qmcp.config.get_settings.cache_clear()
    db_engine._engine = None

    # The server logs every request at INFO through `basicConfig`, which does
    # nothing once the root logger has a handler; a check's output is its
    # verdicts. A failing request still raises in the client.
    root = logging.getLogger()
    saved_level, quiet = root.level, logging.NullHandler()
    root.addHandler(quiet)
    root.setLevel(logging.WARNING)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the throwaway qmcp server did not start")
            time.sleep(0.02)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        if saved_url is None:
            os.environ.pop("QMCP_DATABASE_URL", None)
        else:
            os.environ["QMCP_DATABASE_URL"] = saved_url
        qmcp.config.get_settings.cache_clear()
        db_engine._engine = saved_engine
        root.removeHandler(quiet)
        root.setLevel(saved_level)
