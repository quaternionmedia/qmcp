from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import qmcp.cli as cli


def test_cookbook_list_includes_run() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "list"])

    assert result.exit_code == 0
    assert "simple-plan" in result.output
    assert "run simple-plan" in result.output
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
    assert called["goal"] == "Ship a service"
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
    assert called["goal"] == "Deploy a web service"
    assert called["mcp_url"] == "http://example.com:3333"
    assert called["build"] is True
    assert called["sync"] is True


def test_cookbook_run_rejects_unknown_recipe() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cookbook", "run", "missing"])

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
    assert called == {"host": "0.0.0.0", "port": None, "reload": False}


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
