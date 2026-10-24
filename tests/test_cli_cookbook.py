from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import qmcp.cli as cli


def test_cookbook_list_includes_run() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "list"])

    assert result.exit_code == 0
    assert "simple-plan" in result.output
    assert "approved-deploy" in result.output
    assert "local-qc-gauntlet" in result.output
    assert "run <recipe>" in result.output
    assert "dev <recipe>" in result.output
    assert "docker simple-plan" in result.output
    assert "serve" in result.output


def test_cookbook_simple_plan_dispatches(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    flow_path = repo_root / "examples" / "flows" / "simple_plan.py"

    called: dict[str, object] = {}

    def fake_run_flow_docker(**kwargs):
        called.update(kwargs)

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(cli, "_run_flow_docker", fake_run_flow_docker)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "cookbook",
            "simple-plan",
            "--goal",
            "Ship a service",
            "--mcp-url",
            "http://example.com:3333",
            "--no-build",
            "--metaflow-user",
            "tester",
            "--no-sync",
        ],
    )

    assert result.exit_code == 0
    assert called["repo_root"] == repo_root
    assert called["flow_path"] == flow_path
    assert called["flow_args"] == ["--goal", "Ship a service"]
    assert called["mcp_url"] == "http://example.com:3333"
    assert called["metaflow_user"] == "tester"
    assert called["build"] is False
    assert called["sync"] is False


def test_cookbook_run_dispatches(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    flow_path = repo_root / "examples" / "flows" / "simple_plan.py"

    called: dict[str, object] = {}

    def fake_run_flow_docker(**kwargs):
        called.update(kwargs)

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(cli, "_run_flow_docker", fake_run_flow_docker)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "cookbook",
            "run",
            "simple-plan",
            "--goal",
            "Deploy a web service",
            "--mcp-url",
            "http://example.com:3333",
            "--build",
        ],
    )

    assert result.exit_code == 0
    assert called["repo_root"] == repo_root
    assert called["flow_path"] == flow_path
    assert called["flow_args"] == ["--goal", "Deploy a web service"]
    assert called["mcp_url"] == "http://example.com:3333"
    assert called["build"] is True
    assert called["sync"] is True


def test_cookbook_run_rejects_unknown_recipe() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "run", "missing"])

    assert result.exit_code != 0
    assert "Unknown recipe" in result.output


def test_cookbook_dev_rejects_unknown_recipe() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "dev", "unknown", "--no-start-server"])

    assert result.exit_code != 0
    assert "Unknown recipe" in result.output


def test_cookbook_serve_defaults(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_run_server(host: str | None, port: int | None, reload: bool) -> None:
        called.update({"host": host, "port": port, "reload": reload})

    monkeypatch.setattr(cli, "_run_server", fake_run_server)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "serve"])

    assert result.exit_code == 0
    assert called == {"host": "127.0.0.1", "port": None, "reload": False}


# The server asks nobody who they are, so off loopback anyone who can reach the
# machine can call its tools and answer its human requests. Mutations seen red:
# `cookbook serve`'s default back on 0.0.0.0 -- the defaults test above; the
# warning's `is_loopback` test inverted -- both tests below.


def _serving(monkeypatch) -> dict[str, object]:
    bound: dict[str, object] = {}
    monkeypatch.setattr(cli, "_qmcp_already_serving", lambda host, port: None)
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: bound.update(kwargs))
    return bound


def test_serving_off_loopback_says_what_is_exposed(monkeypatch) -> None:
    bound = _serving(monkeypatch)

    result = CliRunner().invoke(cli.cli, ["cookbook", "serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert bound["host"] == "0.0.0.0"
    assert "Listening on 0.0.0.0: anyone who can reach this machine" in result.stderr


def test_serving_on_loopback_warns_of_nothing(monkeypatch) -> None:
    bound = _serving(monkeypatch)

    result = CliRunner().invoke(cli.cli, ["cookbook", "serve"])

    assert result.exit_code == 0, result.output
    assert bound["host"] == "127.0.0.1"
    assert result.stderr == ""


def test_a_flow_that_starts_a_server_asks_before_exposing_it(monkeypatch) -> None:
    """A container on Linux reaches the host only off loopback, so the server
    a flow starts must be bound there -- and only when asked. Mutation:
    `--server-host` defaulting to 0.0.0.0 again -- the command went on to start
    an exposed server, and the refusal never came."""
    monkeypatch.setattr(cli, "_find_repo_root", lambda: Path(__file__).resolve().parents[1])
    monkeypatch.setattr(cli, "_start_server_process",
                        lambda **kwargs: pytest.fail("a server was started"))

    result = CliRunner().invoke(cli.cli, [
        "cookbook", "dev", "simple-plan", "--goal", "Ship a service",
        "--no-build", "--no-sync"])

    assert result.exit_code != 0
    assert "Pass --server-host 0.0.0.0, and anyone who can reach this machine" in result.output


def test_build_flow_shell_command_sync_modes() -> None:
    command = cli._build_flow_shell_command(
        ["python", "examples/flows/simple_plan.py", "run"],
        sync=True,
    )

    assert command.startswith("uv sync --extra flows && uv run ")

    no_sync_command = cli._build_flow_shell_command(
        ["python", "examples/flows/simple_plan.py", "run"],
        sync=False,
    )

    assert "uv sync --extra flows" not in no_sync_command
    assert "uv run --no-sync" in no_sync_command


def test_cookbook_dev_dispatches(monkeypatch) -> None:
    called: dict[str, object] = {}

    class DummySettings:
        port = 4444

    def fake_run_flow_docker(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(cli, "_run_flow_docker", fake_run_flow_docker)
    monkeypatch.setattr(cli, "_find_repo_root", lambda: Path(__file__).resolve().parents[1])

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "cookbook",
            "dev",
            "simple-plan",
            "--goal",
            "Ship a service",
            "--no-build",
            "--no-sync",
            "--no-start-server",
        ],
    )

    assert result.exit_code == 0
    assert called["flow_args"] == ["--goal", "Ship a service"]
    assert called["mcp_url"] == "http://host.docker.internal:4444"
    assert called["build"] is False
    assert called["sync"] is False


def test_cookbook_run_requires_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "run", "local-qc-gauntlet"])

    assert result.exit_code != 0
    assert "Missing required flow arguments" in result.output
