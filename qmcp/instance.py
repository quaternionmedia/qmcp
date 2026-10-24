"""Which qmcp is answering: a run-file on this machine, and an identity on /health.

Two clones answer `/health` alike, and a session that dials a port it did not
start can measure somebody else's server. The corpus's
`records/DRAFT-monitoring-seam-and-instance-identity.md`, decisions 4 and 5,
is the answer this module holds: a server says which one it is, and leaves a
run-file where a collector on this machine finds it.

- The run-file lives in a machine-scoped directory outside every repository,
  `~/.qmcp/run/` unless `QMCP_RUN_DIR` says otherwise, one file per serving
  process. It carries the port, the resolved database, the start time, the
  session it belongs to (`QMCP_SESSION`, if declared) and the pid, and is
  removed when the server shuts down cleanly.
- `/health` states the same identity, with the database as a fingerprint
  rather than a path: a collector matches the two, and the server does not
  publish a filesystem path to whoever can reach it.

What this cannot do: tell a run-file from a crashed server, which is never
removed, from a live one. A collector matches the run-file against the
answer on its port before believing it -- decision 5 -- and that match is
what tells them apart.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from qmcp import __version__
from qmcp.db.paths import database_file


@lru_cache
def started_at() -> str:
    """When this process started serving, the same for every caller in it."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def resolved_database(database_url: str) -> str:
    """The database a URL names: its file's resolved path, or the URL itself."""
    found = database_file(database_url)
    return str(found.resolve()) if found is not None else database_url


def database_fingerprint(database_url: str) -> str:
    """A short, stable digest of the resolved database, for matching without a path."""
    return hashlib.sha256(resolved_database(database_url).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Identity:
    pid: int
    host: str
    port: int
    database: str
    database_fingerprint: str
    started_at: str
    session: str | None
    version: str

    def public(self) -> dict[str, Any]:
        """What /health states: everything but the database's path."""
        shown = asdict(self)
        del shown["database"], shown["host"]
        return shown


def describe(settings: Any) -> Identity:
    """This process's identity, read from the settings it serves with."""
    return Identity(
        pid=os.getpid(),
        host=settings.host,
        port=settings.port,
        database=resolved_database(settings.database_url),
        database_fingerprint=database_fingerprint(settings.database_url),
        started_at=started_at(),
        session=getattr(settings, "session", None),
        version=__version__,
    )


def run_dir(settings: Any) -> Path:
    configured = getattr(settings, "run_dir", None)
    return Path(configured) if configured else Path.home() / ".qmcp" / "run"


def write_run_file(identity: Identity, directory: Path) -> Path:
    """Write the run-file for this process, whole, and return where."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"qmcp-{identity.pid}.json"
    staged = path.with_suffix(".json.tmp")
    staged.write_text(json.dumps(asdict(identity), indent=2), encoding="utf-8")
    staged.replace(path)
    return path


def remove_run_file(path: Path) -> None:
    path.unlink(missing_ok=True)
