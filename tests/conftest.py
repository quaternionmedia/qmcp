"""Shared test fixtures for pytest."""

import sys
import uuid
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so root-level modules (e.g. qmcp_mcp)
# are importable regardless of the working directory pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _own_vocabulary_overlay(_isolated_settings, tmp_path, monkeypatch):
    """Every test reads an empty per-user phrase file of its own. The real
    one, `~/.qmcp/voice-vocabulary.json`, holds a person's added phrases, and
    a test reading it would pass or fail by what somebody once said."""
    monkeypatch.setenv("QMCP_VOICE_VOCABULARY_PATH",
                       str(tmp_path / "voice-vocabulary.json"))


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the server with isolated in-memory database."""
    import qmcp.config

    # Clear any cached settings FIRST
    qmcp.config.get_settings.cache_clear()

    # Generate unique DB name for THIS test
    unique_name = f"test_{uuid.uuid4().hex}"
    db_url = f"sqlite+aiosqlite:///file:{unique_name}?mode=memory&cache=shared&uri=true"

    # A real Settings, so every field the app reads exists with its default;
    # `_isolated_settings` has already taken the shell and `.env` out of it.
    test_settings = qmcp.config.Settings(
        database_url=db_url, host="127.0.0.1", port=3333, log_level="WARNING")

    import qmcp.db.engine
    import qmcp.server

    # Reset engine to force new connection
    qmcp.db.engine._engine = None
    # `qmcp.server` and `qmcp.db.engine` bind `get_settings` at import, so the
    # stand-in replaces those bindings as well as the module's own.
    for module in (qmcp.config, qmcp.db.engine, qmcp.server):
        monkeypatch.setattr(module, "get_settings", lambda: test_settings)

    # Import create_app AFTER patching
    from fastapi.testclient import TestClient

    from qmcp.server import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    # Cleanup
    qmcp.db.engine._engine = None
