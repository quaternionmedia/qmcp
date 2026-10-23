"""The fixtures every test and every walkthrough page runs under.

`pyproject.toml` loads this with `-p qmcp.testing.plugin` in `addopts`, so
`uv run pytest tests walkthrough` and `qmcp test` both run under it. A
conftest would not reach `walkthrough/`: its pages are collected as doctests
from a directory with no conftest of its own.
"""

from __future__ import annotations

import pytest


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
