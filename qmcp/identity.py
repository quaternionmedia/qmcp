"""Which repository this is, decided once.

Every module that builds an address or a delta's `project` reads it here, so a
fork's addresses carry the fork's owner. An address is what says two readings
are about one thing (`records/DRAFT-a-route-is-an-address.md`), and a guessed
owner would join a fork's work to another organisation's.

It is derived from the repository, not configured, in this order:

1. `QMCP_PROJECT`, for a caller who knows better than the checkout -- a
   container, a test, a repository vendored elsewhere.
2. The `origin` remote, parsed. The ordinary answer, with no setup.
3. `UNKNOWN`, named rather than guessed: a checkout with no remote -- a fresh
   `git init`, a tarball -- is a real state.

The answer is cached for the life of the process, so formatting an address
does not start a subprocess.
"""

from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

UNKNOWN = "unknown/unknown"
"""What this is when the repository cannot say. A value, not an absence:
callers can tell it apart from a real owner and refuse rather than emit it."""

# `git@host:owner/repo.git`, `https://host/owner/repo.git`, and the bare forms.
# Deliberately loose about the host: an address's owner is the account, and
# which forge it is on is not part of it.
_REMOTE = re.compile(
    r"[:/](?P<owner>[^/:]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def _from_remote(start: Path | None = None) -> str | None:
    """`<owner>/<repo>` from the `origin` remote, or None."""
    try:
        done = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(start or Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=5,
        )
    except Exception:                              # noqa: BLE001
        return None
    if done.returncode != 0:
        return None

    found = _REMOTE.search(done.stdout.strip())
    if not found:
        return None
    return f"{found.group('owner')}/{found.group('repo')}"


@lru_cache(maxsize=1)
def this_project() -> str:
    """`<owner>/<repo>` for the repository this harness is running in."""
    declared = os.environ.get("QMCP_PROJECT", "").strip()
    if declared:
        return declared
    return _from_remote() or UNKNOWN


def is_known(project: str | None = None) -> bool:
    """Whether the identity was established rather than fallen back to.

    A caller that must not emit a guessed address checks this. Nothing here
    refuses on its behalf: whether an unknown owner is fatal depends on what is
    being written, and this module is not the place that knows.
    """
    return (project or this_project()) != UNKNOWN


def owner(project: str | None = None) -> str:
    """Just the account part."""
    return (project or this_project()).partition("/")[0]


def forget() -> None:
    """Drop the cache. For tests, and for a process that has changed directory
    into a different checkout."""
    this_project.cache_clear()
