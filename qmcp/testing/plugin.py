"""The fixtures every test and every walkthrough page runs under.

`pyproject.toml` loads this with `-p qmcp.testing.plugin` in `addopts`, so
`uv run pytest tests walkthrough` and `qmcp test` both run under it. A
conftest would not reach `walkthrough/`: its pages are collected as doctests
from a directory with no conftest of its own.

Two guarantees. A test reads only the settings it sets. And nothing reaches
past loopback -- not the test's process, not a Python child it starts -- or
the test fails, naming what was attempted (`qmcp.testing.netguard`). A test
marked `live_anthropic` is about a real service and is let through.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from qmcp.testing import netguard

_LOG = pytest.StashKey[Path]()
_SAVED = pytest.StashKey[dict[str, str | None]]()

ONLINE_MARKERS = ("live_anthropic",)
"""The markers that let a test past the guard."""


def pytest_configure(config: pytest.Config) -> None:
    """Guard the session before collection imports anything, and hand the
    guard to every Python child through the environment."""
    netguard.install()
    log = Path(tempfile.mkdtemp(prefix="qmcp-netguard-")) / "attempts.log"
    log.touch()
    config.stash[_LOG] = log
    config.stash[_SAVED] = {name: os.environ.get(name) for name in netguard.CHILD_VARIABLES}
    guarded = netguard.child_environment(log)
    os.environ.update({name: guarded[name] for name in netguard.CHILD_VARIABLES})


def pytest_unconfigure(config: pytest.Config) -> None:
    for name, value in config.stash.get(_SAVED, {}).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def pytest_collection_finish(session: pytest.Session) -> None:
    """Collection imports every test module; an attempt then belongs to no test."""
    attempts = netguard.taken()
    if attempts:
        raise pytest.UsageError(
            "collecting the suite reached past loopback:\n  " + "\n  ".join(attempts))


@pytest.fixture(autouse=True)
def _nothing_past_loopback(request, pytestconfig):
    """Fail a test that reached past loopback, from this process or a child,
    even when the code under test caught the refusal and carried on."""
    if any(request.node.get_closest_marker(name) for name in ONLINE_MARKERS):
        with netguard.permitted():
            yield
        return
    log = pytestconfig.stash[_LOG]
    start = log.stat().st_size
    yield
    attempts = netguard.taken() + log.read_bytes()[start:].decode("utf-8").splitlines()
    if attempts:
        pytest.fail("reached past loopback:\n  " + "\n  ".join(attempts), pytrace=False)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Every test reads the settings it sets and nothing else: no `QMCP_*`
    variable from the shell that ran pytest, and no `.env` from the directory
    it ran in (`qmcp.testing.isolate_settings`)."""
    from qmcp.config import get_settings
    from qmcp.testing import isolate_settings

    isolate_settings(monkeypatch)
    yield
    get_settings.cache_clear()
