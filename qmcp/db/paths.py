"""Where the configured database actually is on disk.

The setting is a SQLAlchemy URL -- `sqlite+aiosqlite:///./qmcp.db` -- whose
scheme names a driver and whose path is relative to the working directory.
Treated as a filename, the whole string becomes `sqlite+aiosqlite:/qmcp.db`,
which `sqlite3.connect` creates as a new empty database beside the real one;
this function is the one place that reads it correctly.

What this cannot do: answer for a database that is not a file. A URL naming
Postgres, or SQLite's `:memory:`, has no path, and this returns None rather
than inventing one.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def database_file(database_url: str, root: Path | None = None) -> Path | None:
    """The file a SQLite URL points at, or None when it names no file."""
    parsed = urlparse(database_url)
    if not parsed.scheme.startswith("sqlite"):
        return None

    # `sqlite:///./x.db` -> path `/./x.db`; `sqlite:////abs/x.db` -> `//abs/x.db`
    raw = parsed.path
    if not raw or raw in ("/", "/:memory:") or ":memory:" in database_url:
        return None
    raw = raw.lstrip("/") if not raw.startswith("//") else raw[1:]
    if raw.startswith("./"):
        raw = raw[2:]

    path = Path(raw)
    if path.is_absolute():
        return path
    return ((root or Path.cwd()) / path).resolve()
