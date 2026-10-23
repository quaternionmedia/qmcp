"""A test reads the settings it sets, and nothing from the machine running it.

`qmcp.testing.plugin`, loaded from `addopts`, takes every `QMCP_*` variable and
the working directory's `.env` out of what `qmcp.config.Settings` reads, before
every test and every walkthrough page, through `qmcp.testing.isolate_settings`.
These tests put a setting where a developer's machine would and show it does
not arrive.
"""

from __future__ import annotations

import qmcp.config
import qmcp.server
from qmcp.config import Settings, get_settings
from qmcp.testing import isolate_settings


def test_a_dotenv_in_the_working_directory_is_not_read(tmp_path, monkeypatch) -> None:
    # Mutation seen red: the `setitem(Settings.model_config, "env_file", None)`
    # line removed -- `.env` was read and the runtime came back "scripted".
    dotenv = tmp_path / ".env"
    dotenv.write_text("QMCP_CONVERSE_RUNTIME=scripted\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    # The file is one Settings would read when asked to, so the next assertion
    # is about the isolation and not about a file nobody could parse.
    assert Settings(_env_file=dotenv).converse_runtime == "scripted"
    assert get_settings().converse_runtime is None


def test_a_qmcp_variable_in_the_environment_is_removed(monkeypatch) -> None:
    # Mutation seen red: the `delenv` loop removed -- the shell's runtime and
    # port arrived in the settings.
    monkeypatch.setenv("QMCP_CONVERSE_RUNTIME", "scripted")
    monkeypatch.setenv("QMCP_PORT", "1")

    isolate_settings(monkeypatch)

    assert get_settings().converse_runtime is None
    assert get_settings().port == Settings.model_fields["port"].default


def test_the_isolation_is_loaded_for_every_path_pytest_is_given(pytestconfig) -> None:
    # Mutation seen red: `-p qmcp.testing.plugin` removed from `addopts` -- the
    # plugin was not registered, and every test that depends on the vocabulary
    # fixture errored on a missing `_isolated_settings`.
    assert pytestconfig.pluginmanager.has_plugin("qmcp.testing.plugin")


def test_the_client_fixture_settings_are_the_ones_the_server_reads(client) -> None:
    # Mutation seen red: `qmcp.server` dropped from the modules the fixture
    # patches -- the server's binding returned the real settings, port 3141.
    assert qmcp.server.get_settings() is qmcp.config.get_settings()
    assert qmcp.server.get_settings().port == 3333
