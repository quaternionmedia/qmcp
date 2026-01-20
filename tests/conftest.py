"""Shared test fixtures for pytest."""

import uuid

import pytest


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the server with isolated in-memory database."""
    import qmcp.config

    # Clear any cached settings FIRST
    qmcp.config.get_settings.cache_clear()

    # Generate unique DB name for THIS test
    unique_name = f"test_{uuid.uuid4().hex}"
    db_url = f"sqlite+aiosqlite:///file:{unique_name}?mode=memory&cache=shared&uri=true"

    class TestSettings:
        """Test settings with unique in-memory database."""

        def __init__(self, url):
            self.database_url = url
            self.debug = False
            self.host = "127.0.0.1"
            self.port = 3333
            self.log_level = "WARNING"

    test_settings = TestSettings(db_url)

    # Use monkeypatch to replace the function
    monkeypatch.setattr(qmcp.config, "get_settings", lambda: test_settings)

    import qmcp.db.engine

    # Reset engine to force new connection
    qmcp.db.engine._engine = None
    monkeypatch.setattr(qmcp.db.engine, "get_settings", lambda: test_settings)

    # Import create_app AFTER patching
    from fastapi.testclient import TestClient

    from qmcp.server import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    # Cleanup
    qmcp.db.engine._engine = None
