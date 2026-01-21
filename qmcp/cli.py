"""CLI interface for QMCP.

Provides commands for:
- Starting the MCP server
- Listing registered tools
- Development utilities
"""

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import click
import uvicorn

from qmcp import __version__
from qmcp.config import get_settings


def _find_repo_root() -> Path:
    current = Path.cwd().resolve()
    markers = ("pyproject.toml", "docker-compose.flows.yml")
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate
    raise click.ClickException(
        "Could not find repo root with pyproject.toml and docker-compose.flows.yml.",
    )


def _default_metaflow_user() -> str:
    return (
        os.getenv("METAFLOW_USER")
        or os.getenv("USERNAME")
        or os.getenv("USER")
        or "local"
    )


def _default_mcp_url() -> str:
    return os.getenv("MCP_URL", "http://host.docker.internal:3333")


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    click.echo(click.style(f"Running: {' '.join(cmd)}", fg="blue"))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Command failed with exit code {exc.returncode}.",
        ) from exc


def _ensure_docker_available() -> None:
    try:
        subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            "Docker CLI not found. Install Docker Desktop and try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        message = (
            "Docker engine is not reachable. Start Docker Desktop and ensure the "
            "Linux engine is running, then retry."
        )
        if stderr:
            message = f"{message}\nDocker error: {stderr}"
        raise click.ClickException(message) from exc


def _ensure_flow_runner_image(image_tag: str) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Flow-runner image '{image_tag}' not found. Re-run with --build."
        )


def _validate_mcp_url(mcp_url: str) -> str:
    try:
        parsed = urlparse(mcp_url)
    except ValueError as exc:
        raise click.ClickException(f"Invalid MCP URL: {mcp_url}") from exc
    if not parsed.scheme or not parsed.netloc:
        raise click.ClickException(f"Invalid MCP URL: {mcp_url}")
    hostname = parsed.hostname or ""
    if hostname in {"localhost", "127.0.0.1"}:
        raise click.ClickException(
            "MCP URL points at localhost. Use host.docker.internal when running flows in Docker."
        )
    return hostname


def _get_simple_plan_paths() -> tuple[Path, Path]:
    repo_root = _find_repo_root()
    flow_path = repo_root / "examples" / "flows" / "simple_plan.py"
    if not flow_path.exists():
        raise click.ClickException(f"Flow not found at {flow_path}.")
    return repo_root, flow_path


def _run_simple_plan_recipe(
    goal: str,
    mcp_url: str | None,
    build: bool,
    metaflow_user: str | None,
    sync: bool,
) -> None:
    mcp_url = mcp_url or _default_mcp_url()
    metaflow_user = metaflow_user or _default_metaflow_user()
    repo_root, flow_path = _get_simple_plan_paths()

    click.echo(click.style("Running cookbook recipe simple-plan (docker).", fg="green"))
    _run_flow_docker(
        repo_root=repo_root,
        flow_path=flow_path,
        goal=goal,
        mcp_url=mcp_url,
        metaflow_user=metaflow_user,
        build=build,
        sync=sync,
    )


@click.group()
@click.version_option(version=__version__, prog_name="qmcp")
def cli() -> None:
    """QMCP - Model Context Protocol Server.

    A spec-aligned MCP server for tool discovery and invocation.
    """
    pass


@cli.command()
@click.option("--host", "-h", default=None, help="Host to bind to")
@click.option("--port", "-p", default=None, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the MCP server."""
    _run_server(host, port, reload)


@cli.group()
def tools() -> None:
    """Tool management commands."""
    pass


@tools.command("list")
def list_tools() -> None:
    """List all registered tools."""
    # Import to trigger tool registration
    from qmcp.tools import builtin as _  # noqa: F401
    from qmcp.tools import tool_registry

    tools = tool_registry.list_tools()

    if not tools:
        click.echo("No tools registered.")
        return

    click.echo(f"Registered tools ({len(tools)}):\n")

    for tool in tools:
        click.echo(f"  {click.style(tool.name, fg='green', bold=True)}")
        click.echo(f"    {tool.description}")
        if tool.input_schema:
            props = tool.input_schema.get("properties", {})
            if props:
                click.echo(f"    Parameters: {', '.join(props.keys())}")
        click.echo()


@cli.group()
def cookbook() -> None:
    """Cookbook recipes for example flows."""
    pass


@cookbook.command("list")
def list_recipes() -> None:
    """List available cookbook recipes."""
    click.echo("Cookbook recipes:\n")
    click.echo("  simple-plan         Plan -> execute -> review using MCP tools (Docker)")
    click.echo("  run simple-plan     Run simple-plan via the generic runner (Docker)")
    click.echo("  docker simple-plan  Run simple-plan in Docker (explicit)")
    click.echo("  serve               Start the MCP server for Docker flows")


@cookbook.group("docker")
def cookbook_docker() -> None:
    """Run cookbook recipes in Docker."""
    pass


@cookbook.command("serve")
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind to for Docker-based flows.",
)
@click.option("--port", "-p", default=None, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def cookbook_serve(host: str, port: int | None, reload: bool) -> None:
    """Start the MCP server with Docker-friendly defaults."""
    _run_server(host, port, reload)


def _run_server(host: str | None, port: int | None, reload: bool) -> None:
    settings = get_settings()

    actual_host = host or settings.host
    actual_port = port or settings.port

    click.echo(f"Starting QMCP server on {actual_host}:{actual_port}")

    uvicorn.run(
        "qmcp.server:app",
        host=actual_host,
        port=actual_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


def _flow_runner_image_tag(repo_root: Path) -> str:
    return f"{repo_root.name}-flow-runner"


def _build_flow_runner_image(repo_root: Path, image_tag: str) -> None:
    _ensure_docker_available()
    dockerfile_src = repo_root / "docker" / "flows.Dockerfile"
    if not dockerfile_src.exists():
        raise click.ClickException(f"Dockerfile not found at {dockerfile_src}.")

    required_files = ["pyproject.toml", "uv.lock", "README.md"]
    for filename in required_files:
        if not (repo_root / filename).exists():
            raise click.ClickException(f"Required file missing: {repo_root / filename}.")

    with tempfile.TemporaryDirectory(prefix="qmcp-flow-build-") as temp_dir:
        temp_root = Path(temp_dir)
        (temp_root / "docker").mkdir(parents=True, exist_ok=True)
        shutil.copy2(dockerfile_src, temp_root / "docker" / "flows.Dockerfile")
        for filename in required_files:
            shutil.copy2(repo_root / filename, temp_root / filename)
        shutil.copytree(
            repo_root / "qmcp",
            temp_root / "qmcp",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        _run_cmd(
            [
                "docker",
                "build",
                "-f",
                str(temp_root / "docker" / "flows.Dockerfile"),
                "-t",
                image_tag,
                str(temp_root),
            ],
            cwd=temp_root,
        )


def _build_flow_shell_command(flow_args: list[str], sync: bool) -> str:
    uv_run = ["uv", "run"]
    if not sync:
        uv_run.append("--no-sync")
    uv_run.extend(flow_args)
    uv_run_cmd = " ".join(shlex.quote(arg) for arg in uv_run)
    if sync:
        return f"uv sync --extra flows && {uv_run_cmd}"
    return uv_run_cmd


def _run_flow_docker(
    repo_root: Path,
    flow_path: Path,
    goal: str,
    mcp_url: str,
    metaflow_user: str,
    build: bool,
    sync: bool,
) -> None:
    _ensure_docker_available()
    _validate_mcp_url(mcp_url)
    compose_file = repo_root / "docker-compose.flows.yml"
    image_tag = _flow_runner_image_tag(repo_root)
    if build:
        _build_flow_runner_image(repo_root, image_tag)
    else:
        _ensure_flow_runner_image(image_tag)

    flow_rel = flow_path.relative_to(repo_root).as_posix()
    flow_args = [
        "python",
        flow_rel,
        "run",
        "--goal",
        goal,
        "--mcp-url",
        mcp_url,
    ]
    shell_command = _build_flow_shell_command(flow_args, sync=sync)

    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        "--entrypoint",
        "sh",
        "-e",
        "UV_PROJECT_ENVIRONMENT=/tmp/uv-venv",
        "-e",
        f"METAFLOW_USER={metaflow_user}",
        "-e",
        "METAFLOW_HOME=/tmp/metaflow",
        "-e",
        "METAFLOW_DATASTORE_SYSROOT_LOCAL=/tmp/metaflow",
        "-e",
        "FLOW_DB_PATH=/app/.qmcp_devflows.db",
        "-e",
        f"MCP_URL={mcp_url}",
        "flow-runner",
        "-c",
        shell_command,
    ]
    _run_cmd(cmd, cwd=repo_root)


@cookbook.command("simple-plan")
@click.option(
    "--goal",
    default="Deploy a web service",
    show_default=True,
    help="Planning goal to pass into the flow.",
)
@click.option(
    "--mcp-url",
    default=None,
    help="MCP server URL (defaults to host.docker.internal).",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build the flow-runner image before running.",
)
@click.option(
    "--sync/--no-sync",
    default=True,
    help="Sync flow dependencies inside the runner before executing.",
)
@click.option(
    "--metaflow-user",
    default=None,
    help="Override the METAFLOW_USER value for this run.",
)
def run_simple_plan(
    goal: str,
    mcp_url: str | None,
    build: bool,
    sync: bool,
    metaflow_user: str | None,
) -> None:
    """Run the simple planning flow from the cookbook."""
    _run_simple_plan_recipe(
        goal=goal,
        mcp_url=mcp_url,
        build=build,
        metaflow_user=metaflow_user,
        sync=sync,
    )


@cookbook.command("run")
@click.argument("recipe")
@click.option(
    "--goal",
    default="Deploy a web service",
    show_default=True,
    help="Planning goal to pass into the flow.",
)
@click.option(
    "--mcp-url",
    default=None,
    help="MCP server URL (defaults to host.docker.internal).",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build the flow-runner image before running.",
)
@click.option(
    "--sync/--no-sync",
    default=True,
    help="Sync flow dependencies inside the runner before executing.",
)
@click.option(
    "--metaflow-user",
    default=None,
    help="Override the METAFLOW_USER value for this run.",
)
def run_cookbook_recipe(
    recipe: str,
    goal: str,
    mcp_url: str | None,
    build: bool,
    sync: bool,
    metaflow_user: str | None,
) -> None:
    """Run a cookbook recipe in Docker."""
    normalized = recipe.lower().replace("_", "-")
    if normalized != "simple-plan":
        raise click.ClickException(
            "Unknown recipe. Available recipes: simple-plan",
        )
    _run_simple_plan_recipe(
        goal=goal,
        mcp_url=mcp_url,
        build=build,
        metaflow_user=metaflow_user,
        sync=sync,
    )


@cookbook_docker.command("simple-plan")
@click.option(
    "--goal",
    default="Deploy a web service",
    show_default=True,
    help="Planning goal to pass into the flow.",
)
@click.option(
    "--mcp-url",
    default=None,
    help="MCP server URL (defaults to host.docker.internal).",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build the flow-runner image before running.",
)
@click.option(
    "--sync/--no-sync",
    default=True,
    help="Sync flow dependencies inside the runner before executing.",
)
@click.option(
    "--metaflow-user",
    default=None,
    help="Override the METAFLOW_USER value for this run.",
)
def run_simple_plan_docker(
    goal: str,
    mcp_url: str | None,
    build: bool,
    sync: bool,
    metaflow_user: str | None,
) -> None:
    """Run the simple planning flow in Docker."""
    _run_simple_plan_recipe(
        goal=goal,
        mcp_url=mcp_url,
        build=build,
        metaflow_user=metaflow_user,
        sync=sync,
    )


@cli.command()
def info() -> None:
    """Show server configuration."""
    settings = get_settings()

    click.echo("QMCP Configuration:\n")
    click.echo(f"  Host:     {settings.host}")
    click.echo(f"  Port:     {settings.port}")
    click.echo(f"  Debug:    {settings.debug}")
    click.echo(f"  Log Level: {settings.log_level}")
    click.echo(f"  Database: {settings.database_url}")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--coverage", is_flag=True, help="Run with coverage report")
@click.option("--clean", is_flag=True, default=True, help="Clean database before tests (default: True)")
@click.argument("test_path", required=False)
def test(verbose: bool, coverage: bool, clean: bool, test_path: str | None) -> None:
    """Run the test suite with automatic setup/teardown.

    Optionally specify a test path like 'tests/test_hitl.py' or
    'tests/test_server.py::TestHealthEndpoint'.
    """
    import subprocess
    import sys
    from pathlib import Path

    # Setup: Clean database file if requested
    if clean:
        db_file = Path("qmcp.db")
        if db_file.exists():
            db_file.unlink()
            click.echo(click.style("✓ Cleaned qmcp.db", fg="yellow"))

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=qmcp", "--cov-report=term-missing"])

    if test_path:
        cmd.append(test_path)

    click.echo(click.style(f"Running: {' '.join(cmd)}", fg="blue"))
    click.echo()

    # Run tests
    result = subprocess.run(cmd)

    # Teardown: Clean database file after tests
    if clean:
        db_file = Path("qmcp.db")
        if db_file.exists():
            db_file.unlink()
            click.echo()
            click.echo(click.style("✓ Cleaned qmcp.db after tests", fg="yellow"))

    # Exit with pytest's exit code
    sys.exit(result.returncode)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
