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
import sys
import tempfile
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    description: str
    flow_rel: str
    required_flags: tuple[str, ...] = ()


def _recipe_specs(repo_root: Path) -> dict[str, RecipeSpec]:
    return {
        "simple-plan": RecipeSpec(
            name="simple-plan",
            description="Plan -> execute -> review using MCP tools",
            flow_rel="examples/flows/simple_plan.py",
        ),
        "approved-deploy": RecipeSpec(
            name="approved-deploy",
            description="HITL approval workflow for deployments",
            flow_rel="examples/flows/approved_deploy.py",
            required_flags=("--service",),
        ),
        "local-agent-chain": RecipeSpec(
            name="local-agent-chain",
            description="Local LLM plan -> review -> refine chain",
            flow_rel="examples/flows/local_agent_chain.py",
            required_flags=("--goal",),
        ),
        "local-qc-gauntlet": RecipeSpec(
            name="local-qc-gauntlet",
            description="Local LLM QC checklist + tasks + gate",
            flow_rel="examples/flows/local_qc_gauntlet.py",
            required_flags=("--change-summary",),
        ),
        "local-release-notes": RecipeSpec(
            name="local-release-notes",
            description="Local LLM release notes + doc updates",
            flow_rel="examples/flows/local_release_notes.py",
            required_flags=("--change-summary",),
        ),
        "council-deliberation": RecipeSpec(
            name="council-deliberation",
            description="Multi-agent council deliberation for decisions",
            flow_rel="examples/flows/council_deliberation.py",
            required_flags=("--question",),
        ),
        # Compound recipes
        "qc-release": RecipeSpec(
            name="qc-release",
            description="QC gauntlet + release notes compound pipeline",
            flow_rel="examples/flows/qc_release.py",
            required_flags=("--change-summary",),
        ),
        "plan-council": RecipeSpec(
            name="plan-council",
            description="Plan + council deliberation + refinement",
            flow_rel="examples/flows/plan_council.py",
            required_flags=("--goal",),
        ),
        "change-impact": RecipeSpec(
            name="change-impact",
            description="Full change impact analysis pipeline",
            flow_rel="examples/flows/change_impact.py",
            required_flags=("--change-summary",),
        ),
    }


def _resolve_recipe(repo_root: Path, recipe: str) -> RecipeSpec:
    normalized = recipe.lower().replace("_", "-")
    spec = _recipe_specs(repo_root).get(normalized)
    if not spec:
        raise click.ClickException(
            "Unknown recipe. Available recipes: "
            + ", ".join(sorted(_recipe_specs(repo_root).keys()))
        )
    flow_path = repo_root / spec.flow_rel
    if not flow_path.exists():
        raise click.ClickException(f"Flow not found at {flow_path}.")
    return spec


def _flag_present(args: list[str], flag: str) -> bool:
    if flag in args:
        return True
    prefix = f"{flag}="
    return any(arg.startswith(prefix) for arg in args)


def _extract_flag_value(args: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _ensure_required_flags(flow_args: list[str], required_flags: tuple[str, ...]) -> None:
    missing = [flag for flag in required_flags if not _flag_present(flow_args, flag)]
    if missing:
        raise click.ClickException(
            "Missing required flow arguments: " + ", ".join(missing)
        )


def _default_flow_mcp_url(server_host: str, server_port: int) -> str:
    host = server_host or "0.0.0.0"
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        host = "host.docker.internal"
    return f"http://{host}:{server_port}"


def _server_health_url(server_host: str, server_port: int) -> str:
    host = server_host or "127.0.0.1"
    if host in {"0.0.0.0", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{server_port}/health"


def _is_server_healthy(health_url: str) -> bool:
    try:
        import httpx
    except ImportError as exc:
        raise click.ClickException("httpx is required to run health checks.") from exc
    try:
        response = httpx.get(health_url, timeout=1.0)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _wait_for_server(health_url: str, timeout_seconds: float, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise click.ClickException(
                "MCP server process exited before becoming healthy."
            )
        if _is_server_healthy(health_url):
            return
        time.sleep(0.25)
    raise click.ClickException(
        f"MCP server did not become healthy within {timeout_seconds:.1f}s at {health_url}."
    )


def _start_server_process(
    repo_root: Path,
    host: str,
    port: int,
    reload: bool,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "qmcp",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    click.echo(click.style(f"Starting MCP server: {' '.join(cmd)}", fg="blue"))
    return subprocess.Popen(cmd, cwd=repo_root)


def _stop_server_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


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
        flow_args=["--goal", goal],
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
    """Start the MCP server.

    FOR A DEV SERVER LEFT RUNNING, START IT AS A MODULE:

        uv run python -m qmcp serve

    not `uv run qmcp serve`. The console script is `Scripts/qmcp.exe`, and
    Windows locks a running executable -- so any `uv sync` that reinstalls the
    package fails with "The process cannot access the file because it is being
    used by another process" until the server is stopped. Running the module
    never opens that file, and `uv sync` works with the server up.
    """
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
    repo_root = _find_repo_root()
    click.echo("Cookbook recipes:\n")
    for name, spec in _recipe_specs(repo_root).items():
        click.echo(f"  {name:<18} {spec.description} (Docker)")
    click.echo("  run <recipe>        Run a recipe via the generic runner (Docker)")
    click.echo("  dev <recipe>        Start server + run a recipe (Docker)")
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
    flow_args: list[str],
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
    args = ["python", flow_rel, "run"]
    args.extend(flow_args)
    if mcp_url and not _flag_present(args, "--mcp-url"):
        args.extend(["--mcp-url", mcp_url])
    shell_command = _build_flow_shell_command(args, sync=sync)

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


@cookbook.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("recipe")
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
@click.pass_context
def run_cookbook_recipe(
    ctx: click.Context,
    recipe: str,
    mcp_url: str | None,
    build: bool,
    sync: bool,
    metaflow_user: str | None,
) -> None:
    """Run a cookbook recipe in Docker."""
    repo_root = _find_repo_root()
    spec = _resolve_recipe(repo_root, recipe)
    flow_path = repo_root / spec.flow_rel
    flow_args = list(ctx.args)
    mcp_from_args = _extract_flag_value(flow_args, "--mcp-url")
    mcp_url = mcp_url or mcp_from_args or _default_mcp_url()
    metaflow_user = metaflow_user or _default_metaflow_user()
    if spec.name == "simple-plan" and not _flag_present(flow_args, "--goal"):
        flow_args.extend(["--goal", "Deploy a web service"])
    _ensure_required_flags(flow_args, spec.required_flags)
    _run_flow_docker(
        repo_root=repo_root,
        flow_path=flow_path,
        flow_args=flow_args,
        mcp_url=mcp_url,
        metaflow_user=metaflow_user,
        build=build,
        sync=sync,
    )


@cookbook.command(
    "dev",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("recipe", default="simple-plan", required=False)
@click.option(
    "--mcp-url",
    default=None,
    help="Override the MCP URL passed to the flow.",
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
@click.option(
    "--start-server/--no-start-server",
    default=True,
    help="Start the MCP server before running the flow.",
)
@click.option(
    "--server-host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind the MCP server for Docker access.",
)
@click.option(
    "--server-port",
    default=None,
    type=int,
    help="Port to bind the MCP server.",
)
@click.option(
    "--server-reload",
    is_flag=True,
    help="Enable auto-reload for the MCP server.",
)
@click.option(
    "--server-wait",
    default=15.0,
    show_default=True,
    type=float,
    help="Seconds to wait for the MCP server health check.",
)
@click.option(
    "--keep-server",
    is_flag=True,
    help="Leave the MCP server running after the flow completes.",
)
@click.pass_context
def cookbook_dev(
    ctx: click.Context,
    recipe: str,
    mcp_url: str | None,
    build: bool,
    sync: bool,
    metaflow_user: str | None,
    start_server: bool,
    server_host: str,
    server_port: int | None,
    server_reload: bool,
    server_wait: float,
    keep_server: bool,
) -> None:
    """Start the MCP server and run a cookbook recipe in Docker."""
    repo_root = _find_repo_root()
    spec = _resolve_recipe(repo_root, recipe)
    flow_path = repo_root / spec.flow_rel

    settings = get_settings()
    server_port = server_port or settings.port

    if start_server and server_host in {"127.0.0.1", "localhost"}:
        raise click.ClickException(
            "Docker flows cannot reach a server bound to localhost. Use --server-host 0.0.0.0."
        )

    health_url = _server_health_url(server_host, server_port)
    server_process: subprocess.Popen | None = None
    started_server = False
    flow_args = list(ctx.args)
    _ensure_required_flags(flow_args, spec.required_flags)
    mcp_from_args = _extract_flag_value(flow_args, "--mcp-url")
    try:
        if start_server:
            if _is_server_healthy(health_url):
                click.echo(click.style("MCP server already running.", fg="yellow"))
            else:
                server_process = _start_server_process(
                    repo_root=repo_root,
                    host=server_host,
                    port=server_port,
                    reload=server_reload,
                )
                started_server = True
                _wait_for_server(health_url, server_wait, server_process)

        if spec.name == "simple-plan" and not _flag_present(flow_args, "--goal"):
            flow_args.extend(["--goal", "Deploy a web service"])
        flow_mcp_url = mcp_url or mcp_from_args or _default_flow_mcp_url(
            server_host, server_port
        )
        _run_flow_docker(
            repo_root=repo_root,
            flow_path=flow_path,
            flow_args=flow_args,
            mcp_url=flow_mcp_url,
            metaflow_user=metaflow_user or _default_metaflow_user(),
            build=build,
            sync=sync,
        )
    finally:
        if started_server and not keep_server and server_process is not None:
            _stop_server_process(server_process)


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


@cli.group()
def council() -> None:
    """Council topology management commands."""
    pass


@council.command("create")
@click.option(
    "--name",
    required=True,
    help="Name for the council topology.",
)
@click.option(
    "--description",
    default="Council for multi-perspective deliberation",
    help="Description of the council's purpose.",
)
@click.option(
    "--max-rounds",
    default=5,
    type=int,
    help="Maximum deliberation rounds before arbiter decides.",
)
@click.option(
    "--consensus-threshold",
    default=0.67,
    type=float,
    help="Proportion required for consensus (0.5=majority, 0.67=supermajority, 1.0=unanimous).",
)
@click.option(
    "--arbiter-override/--no-arbiter-override",
    default=True,
    help="Allow arbiter to make final decision if no consensus.",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format.",
)
def council_create(
    name: str,
    description: str,
    max_rounds: int,
    consensus_threshold: float,
    arbiter_override: bool,
    output: str,
) -> None:
    """Create a new council topology configuration.

    Creates a council with 9 specialized agent roles:
    - Council Manager (Arbiter): Facilitates and decides
    - Relatable Storyteller: Frames issues narratively
    - Infinite Dreamer: Explores possibilities
    - Pragmatic Strategist: Focuses on implementation
    - Sanity Check: Validates feasibility
    - Tidy Archivist: Maintains context
    - Brutal Efficist: Demands efficiency
    - Eager Accomplisher: Drives completion
    - Technical Reflector: Provides technical analysis
    """
    from qmcp.agentframework import CouncilConfig, Topology, TopologyType

    config = CouncilConfig(
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
        arbiter_can_override=arbiter_override,
    )

    topology = Topology(
        name=name,
        description=description,
        topology_type=TopologyType.COUNCIL,
        config=config.model_dump(),
    )

    if output == "json":
        import json

        click.echo(json.dumps(topology.model_dump(), indent=2, default=str))
    elif output == "yaml":
        try:
            import yaml

            click.echo(yaml.dump(topology.model_dump(), default_flow_style=False))
        except ImportError:
            click.echo("PyYAML not installed. Falling back to JSON.")
            import json

            click.echo(json.dumps(topology.model_dump(), indent=2, default=str))
    else:
        click.echo(f"\n{click.style('Council Topology Created', fg='green', bold=True)}\n")
        click.echo(f"  Name:                {topology.name}")
        click.echo(f"  Type:                {topology.topology_type.value}")
        click.echo(f"  Description:         {topology.description}")
        click.echo(f"\n  {click.style('Configuration:', bold=True)}")
        click.echo(f"    Max Rounds:        {config.max_rounds}")
        click.echo(f"    Consensus:         {config.consensus_threshold:.0%}")
        click.echo(f"    Arbiter Override:  {config.arbiter_can_override}")
        click.echo(f"    Deliberation:      {config.deliberation_style}")
        click.echo(f"\n  {click.style('Council Members:', bold=True)}")
        members = [
            ("arbiter", "Council Manager", "Facilitates, synthesizes, decides"),
            ("storyteller", "Relatable Storyteller", "Frames in narrative form"),
            ("dreamer", "Infinite Dreamer", "Explores possibilities"),
            ("strategist", "Pragmatic Strategist", "Implementation focus"),
            ("sanity_check", "Sanity Check", "Validates feasibility"),
            ("archivist", "Tidy Archivist", "Maintains context"),
            ("efficist", "Brutal Efficist", "Demands efficiency"),
            ("accomplisher", "Eager Accomplisher", "Drives completion"),
            ("reflector", "Technical Reflector", "Technical analysis"),
        ]
        for slot, role, desc in members:
            click.echo(f"    {slot:<14} {role:<22} {desc}")


@council.command("run")
@click.option(
    "--question",
    "-q",
    required=True,
    help="The question for the council to deliberate.",
)
@click.option(
    "--context",
    "-c",
    default="",
    help="Additional context for the deliberation.",
)
@click.option(
    "--max-rounds",
    default=2,
    type=int,
    help="Maximum deliberation rounds (default: 2 for speed).",
)
@click.option(
    "--consensus-threshold",
    default=0.67,
    type=float,
    help="Proportion required for consensus.",
)
@click.option(
    "--llm-base-url",
    default=None,
    help="OpenAI-compatible base URL for the LLM.",
)
@click.option(
    "--llm-model",
    default=None,
    help="Model name to use.",
)
@click.option(
    "--llm-api-key",
    default=None,
    help="API key if required.",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build the flow-runner image before running.",
)
@click.option(
    "--sync/--no-sync",
    default=True,
    help="Sync dependencies inside the runner.",
)
def council_run(
    question: str,
    context: str,
    max_rounds: int,
    consensus_threshold: float,
    llm_base_url: str | None,
    llm_model: str | None,
    llm_api_key: str | None,
    build: bool,
    sync: bool,
) -> None:
    """Run a council deliberation flow.

    Executes the council_deliberation.py flow with the specified parameters.
    The council will deliberate on the question until consensus is reached
    or max rounds are exhausted.
    """
    repo_root = _find_repo_root()
    flow_path = repo_root / "examples" / "flows" / "council_deliberation.py"

    if not flow_path.exists():
        raise click.ClickException(f"Council flow not found at {flow_path}")

    flow_args = ["--question", question]
    if context:
        flow_args.extend(["--context", context])
    flow_args.extend(["--max-rounds", str(max_rounds)])
    flow_args.extend(["--consensus-threshold", str(consensus_threshold)])

    if llm_base_url:
        flow_args.extend(["--llm-base-url", llm_base_url])
    if llm_model:
        flow_args.extend(["--llm-model", llm_model])
    if llm_api_key:
        flow_args.extend(["--llm-api-key", llm_api_key])

    click.echo(click.style("Running council deliberation...", fg="green"))
    click.echo(f"  Question: {question}")
    if context:
        click.echo(f"  Context: {context}")
    click.echo(f"  Max rounds: {max_rounds}")
    click.echo(f"  Consensus: {consensus_threshold:.0%}")
    click.echo()

    _run_flow_docker(
        repo_root=repo_root,
        flow_path=flow_path,
        flow_args=flow_args,
        mcp_url=_default_mcp_url(),
        metaflow_user=_default_metaflow_user(),
        build=build,
        sync=sync,
    )


@council.command("members")
def council_members() -> None:
    """List council member roles and their responsibilities."""
    click.echo(f"\n{click.style('Council Member Roles', fg='green', bold=True)}\n")

    members = [
        (
            "arbiter",
            "Council Manager",
            "COORDINATOR",
            "Facilitates discussion, synthesizes viewpoints, makes final decisions",
        ),
        (
            "storyteller",
            "Relatable Storyteller",
            "SPECIALIST",
            "Frames technical issues as human stories, uses analogies",
        ),
        (
            "dreamer",
            "Infinite Dreamer",
            "SPECIALIST",
            "Explores possibilities without constraint, blue-sky thinking",
        ),
        (
            "strategist",
            "Pragmatic Strategist",
            "PLANNER",
            "Focuses on practical implementation, resources, timelines",
        ),
        (
            "sanity_check",
            "Sanity Check",
            "REVIEWER",
            "Devil's advocate, finds edge cases, risks, and problems",
        ),
        (
            "archivist",
            "Tidy Archivist",
            "SPECIALIST",
            "References past decisions, maintains institutional memory",
        ),
        (
            "efficist",
            "Brutal Efficist",
            "CRITIC",
            "Cuts through complexity, demands efficiency, eliminates waste",
        ),
        (
            "accomplisher",
            "Eager Accomplisher",
            "EXECUTOR",
            "Drives toward action, breaks blockers, focuses on shipping",
        ),
        (
            "reflector",
            "Technical Reflector",
            "SPECIALIST",
            "Deep technical analysis, architecture, long-term implications",
        ),
    ]

    for slot, name, role, desc in members:
        click.echo(f"  {click.style(slot, fg='cyan', bold=True):<20}")
        click.echo(f"    Name: {name}")
        click.echo(f"    Role: {role}")
        click.echo(f"    {desc}")
        click.echo()


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


@cli.group()
def db() -> None:
    """Database backup, verification and restore.

    Nothing here migrates. `qmcp db upgrade` is alembic's, and a backup is not
    a migration: restoring an old file restores an old schema.
    """
    pass


def _configured_database() -> Path:
    """The database file the settings point at, or exit saying why not."""
    from qmcp.db.paths import database_file

    settings = get_settings()
    found = database_file(settings.database_url)
    if found is None:
        raise SystemExit(
            f"{settings.database_url}: names no file on disk, so there is "
            f"nothing to copy. A memory or server database is backed up by "
            f"whatever runs it."
        )
    return found


def _show(checked) -> None:
    click.echo(f"  integrity  {checked.integrity}")
    for name, count in sorted(checked.tables.items()):
        click.echo(f"  {name:<24} {count} row(s)")


@db.command("backup")
@click.option("--source", type=click.Path(path_type=Path), default=None,
              help="database to copy (default: the configured one)")
@click.option("--to", "destination", type=click.Path(path_type=Path), default=None,
              help="write here instead of the timestamped default")
def db_backup(source: Path | None, destination: Path | None) -> None:
    """Take a verified copy of the database, with the server still running."""
    from qmcp.db.backup import compare, take

    origin = source or _configured_database()
    click.echo(f"source      {origin}")
    target, checked = take(origin, destination)
    click.echo(f"backup      {target}")
    _show(checked)

    problems = compare(origin, target)
    for problem in problems:
        click.echo(click.style(f"  ! {problem}", fg="red"))
    if problems:
        raise SystemExit(f"{len(problems)} difference(s) between source and copy.")
    click.echo(click.style("verified: same tables, same row counts.", fg="green"))
    click.echo("This does NOT mean the schema is current -- a backup preserves "
               "whatever shape it copied.")


@db.command("backups")
@click.option("--source", type=click.Path(path_type=Path), default=None)
def db_backups(source: Path | None) -> None:
    """List backups of this database, newest first."""
    from qmcp.db.backup import listing

    origin = source or _configured_database()
    found = listing(origin)
    if not found:
        click.echo(f"No backups of {origin.name}. `qmcp db backup` takes one.")
        return
    click.echo(f"{len(found)} backup(s) of {origin.name}, newest first:")
    for path in found:
        click.echo(f"  {path.name:<40} {path.stat().st_size:>10} bytes")


@db.command("verify")
@click.argument("path", type=click.Path(path_type=Path), required=False)
def db_verify(path: Path | None) -> None:
    """Open a database and report what was established about it."""
    from qmcp.db.backup import verify

    target = path or _configured_database()
    checked = verify(target)
    click.echo(f"file        {target}")
    _show(checked)
    if not checked.ok:
        raise SystemExit(f"{target}: does not verify ({checked.reason or checked.integrity}).")
    click.echo(click.style("verified.", fg="green"))
    click.echo("An intact database can still hold a schema the code has moved "
               "past -- `qmcp db current` reads that.")


@db.command("restore")
@click.argument("backup", type=click.Path(exists=True, path_type=Path))
@click.option("--to", "destination", type=click.Path(path_type=Path), default=None)
@click.confirmation_option(prompt="Replace the database with this backup?")
def db_restore(backup: Path, destination: Path | None) -> None:
    """Put a backup back. What is there now is backed up first, always."""
    from qmcp.db.backup import restore

    target = destination or _configured_database()
    displaced, checked = restore(backup, target)
    if displaced:
        click.echo(f"displaced   {displaced}   (the state that was there)")
    click.echo(f"restored    {target}")
    _show(checked)
    click.echo(click.style("restored.", fg="green"))

@db.command("drift")
@click.argument("path", type=click.Path(path_type=Path), required=False)
def db_drift(path: Path | None) -> None:
    """Does this database have the shape the code expects?

    The question nothing asked before a request did. An intact database and a
    current one are different facts.
    """
    from qmcp.db.schema import drift

    target = path or _configured_database()
    found = drift(target)
    click.echo(f"database    {target}")
    if found.clean:
        click.echo(click.style("no drift: every model table and column is there.", fg="green"))
        click.echo("This compares names, not types or constraints -- see "
                   "qmcp/db/schema.py for what it cannot see.")
        return
    for line in found.lines():
        click.echo(click.style(f"  ! {line}", fg="red"))
    raise SystemExit(
        f"{len(found.lines())} difference(s). `qmcp db upgrade` applies pending "
        f"migrations; a difference that survives one is a missing migration."
    )


def _alembic(*args: str) -> int:
    """Run alembic in-process, so its exit status is its own."""
    from alembic.config import main as alembic_main

    try:
        alembic_main(argv=list(args), prog="qmcp db")
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


@db.command("current")
def db_current() -> None:
    """The revision this database is stamped at."""
    raise SystemExit(_alembic("current", "--verbose"))


@db.command("history")
def db_history() -> None:
    """The migration chain."""
    raise SystemExit(_alembic("history", "--indicate-current"))


@db.command("upgrade")
@click.argument("revision", default="head")
def db_upgrade(revision: str) -> None:
    """Apply pending migrations.

    Take a backup first. `qmcp db backup` does it with the server running, and
    a migration that fails part-way leaves the database changed and its
    revision unmoved -- which has happened here.
    """
    raise SystemExit(_alembic("upgrade", revision))


@db.command("stamp")
@click.argument("revision", default="head")
@click.confirmation_option(
    prompt="Stamping asserts the database already has that shape, without checking. Continue?"
)
def db_stamp(revision: str) -> None:
    """Record a revision without running it.

    An assertion, not an operation: it claims the schema is already there.
    `qmcp db drift` is what checks the claim.
    """
    raise SystemExit(_alembic("stamp", revision))


@cli.command("dashboard")
@click.option("--database", type=click.Path(path_type=Path), default=None,
              help="read this database instead of the configured one")
@click.option("--project", default=None,
              help="owner/repo this server's rows belong to")
@click.option("--recent", default=10, show_default=True, help="rows to list")
@click.option("--json", "as_json", is_flag=True, help="emit the view as data")
def dashboard(database: Path | None, project: str | None, recent: int,
              as_json: bool) -> None:
    """qmcp's own view of what it has run.

    Reads the database directly, not the HTTP API: a dashboard that needed the
    server up could not tell you why the server is down.

    Put it beside dossier's -- `dossier dashboard` in another pane. The two show
    different halves of one dataset, joined by the address on every row here.
    """
    import json as _json

    from qmcp.dashboard import DEFAULT_PROJECT, build, render, to_dict

    target = database or _configured_database()
    view = build(target, project or DEFAULT_PROJECT, recent)
    if as_json:
        click.echo(_json.dumps(to_dict(view), indent=2))
        return
    click.echo(render(view))


@cli.group("human")
def human() -> None:
    """The human-in-the-loop queue: what is waiting on a person.

    Reads and writes the database directly rather than through the HTTP API,
    for the reason the dashboard does: a queue you cannot read when the server
    is down is a queue you cannot act on, and the server being down is when
    somebody most wants to know what is outstanding.
    """


@human.command("list")
@click.option("--database", type=click.Path(path_type=Path), default=None)
@click.option("--all", "show_all", is_flag=True,
              help="include requests that have already been answered")
def human_list(database: Path | None, show_all: bool) -> None:
    """What is waiting on a person, oldest first."""
    from sqlmodel import Session, create_engine, select

    from qmcp.db.models import HumanRequest, HumanResponse

    engine = create_engine(f"sqlite:///{Path(database or _configured_database()).as_posix()}")
    with Session(engine) as session:
        requests = session.exec(
            select(HumanRequest).order_by(HumanRequest.created_at)).all()
        answers = {r.request_id: r for r in session.exec(select(HumanResponse)).all()}

        shown = 0
        for request in requests:
            reply = answers.get(request.id)
            if reply is not None and not show_all:
                continue
            shown += 1
            mark = "[?]" if reply is None else "[=]"
            click.echo(f"  {mark} {request.id}")
            click.echo(f"      {request.prompt}")
            if request.options:
                click.echo(f"      options: {', '.join(request.options)}")
            if reply is not None:
                click.echo(f"      answered: {reply.response}"
                           + (f"  ({reply.responded_by})" if reply.responded_by else ""))
            click.echo("")

        if not shown:
            click.echo("  Nothing is waiting on a person."
                       + ("" if show_all else "  (--all includes answered ones.)"))
            return
        click.echo(f"  {shown} waiting."
                   if not show_all else f"  {shown} request(s).")


@human.command("respond")
@click.argument("request_id")
@click.argument("response")
@click.option("--database", type=click.Path(path_type=Path), default=None)
@click.option("--by", default=None, help="who answered")
def human_respond(request_id: str, response: str, database: Path | None,
                  by: str | None) -> None:
    """Answer one request. This is a person acting, and it is recorded as one.

    A response does not resolve whatever the request was about. It records that
    somebody was asked and answered, which is a different fact and the only one
    this can establish.
    """
    from datetime import UTC, datetime

    from sqlmodel import Session, create_engine, select

    from qmcp.db.models import HumanRequest, HumanRequestStatus, HumanResponse

    engine = create_engine(f"sqlite:///{Path(database or _configured_database()).as_posix()}")
    with Session(engine) as session:
        request = session.get(HumanRequest, request_id)
        if request is None:
            raise SystemExit(f"{request_id}: no such request. `qmcp human list` shows them.")
        if request.options and response not in request.options:
            raise SystemExit(
                f"{response!r} is not one of {', '.join(request.options)}. "
                f"A request that named its options is answered with one of them."
            )
        existing = session.exec(
            select(HumanResponse).where(HumanResponse.request_id == request_id)).first()
        if existing is not None:
            raise SystemExit(
                f"{request_id} was already answered {existing.response!r}. "
                f"Nothing here overwrites a person's answer."
            )

        session.add(HumanResponse(request_id=request_id, response=response,
                                  responded_by=by))
        request.status = HumanRequestStatus.RESPONDED
        session.add(request)
        session.commit()

    click.echo(f"  {request_id} answered {response!r}.")
    click.echo("  The unit of work behind it moves to `planning`: somebody has")
    click.echo("  looked. It does not move further, because being asked is not")
    click.echo("  the same as the work being done.")


@cli.command("selfcheck")
@click.option("--database", type=click.Path(path_type=Path), default=None,
              help="record the invocations here instead of the configured database")
@click.option("--project", default=None, help="owner/repo these rows belong to")
@click.option("--deltas", "as_deltas", is_flag=True,
              help="emit the failures as delta payloads instead of a report")
@click.option("--json", "as_json", is_flag=True, help="emit the run as data")
@click.option("--ask/--no-ask", default=True, show_default=True,
              help="raise a human request for each failing check")
def selfcheck(database: Path | None, project: str | None, as_deltas: bool,
              as_json: bool, ask: bool) -> None:
    """Run this repository's own gates, and record the run like any other.

    Each check is a real subprocess against this working tree, written to the
    database as a `ToolInvocation` -- the same row the server writes and the
    same row `qmcp dashboard` reads back. A failing check becomes a unit of
    work; a passing one becomes nothing, because a green gate is not work.

    Pair it with the control panel:

        uv run qmcp selfcheck --deltas > deltas.json
        uv run qmcp dashboard --json > harness.json
        # then, in dossier
        uv run dossier deltas ingest deltas.json --write
        uv run dossier harness ingest harness.json --write
    """
    import json as _json
    import tempfile

    from sqlmodel import Session, SQLModel, create_engine, select

    from qmcp.dashboard import DEFAULT_PROJECT
    from qmcp.db.models import HumanRequest, HumanResponse
    from qmcp.selfcheck import checks, render, run_check, to_delta

    repo = Path(__file__).resolve().parent.parent
    owner_repo = project or DEFAULT_PROJECT
    target = database or _configured_database()

    engine = create_engine(f"sqlite:///{Path(target).as_posix()}")
    SQLModel.metadata.create_all(engine)

    # The captured run goes to a temporary directory. Writing it into the
    # repository would make a self-check dirty the tree it is checking, which
    # is the measurement disturbing its own subject.
    capture_dir = Path(tempfile.mkdtemp(prefix="qmcp-selfcheck-"))

    findings = []
    with Session(engine) as session:
        for check in checks(capture_dir):
            finding, invocation = run_check(check, repo, owner_repo)
            session.add(invocation)
            findings.append(finding)
        session.commit()

        # A question is raised once per failing check and not once per run.
        # Asking again on every run would fill the queue with the same question
        # and bury the one somebody had not answered yet.
        answered = {}
        for finding in findings:
            if finding.ok:
                continue
            from qmcp.selfcheck import ask as ask_about
            request = ask_about(finding, owner_repo)
            existing = session.get(HumanRequest, request.id)
            if existing is None and ask:
                session.add(request)
            reply = session.exec(
                select(HumanResponse).where(HumanResponse.request_id == request.id)
            ).first()
            answered[finding.check] = reply is not None
        session.commit()

    if as_deltas:
        payloads = [to_delta(f, owner_repo, answered=answered.get(f.check, False))
                    for f in findings if not f.ok]
        click.echo(_json.dumps(payloads, indent=2))
        return

    if as_json:
        click.echo(_json.dumps({
            "schema": 1,
            "project": owner_repo,
            "findings": [
                {"check": f.check, "ok": f.ok, "address": f.address,
                 "duration_ms": f.duration_ms, "detail": f.detail}
                for f in findings
            ],
        }, indent=2))
        return

    click.echo(render(findings, owner_repo))


@cli.command("deltas")
@click.option("--project", default=None, help="owner/repo these belong to")
@click.option("--pipeline", default="change_impact", show_default=True,
              help="which cookbook pipeline's steps to emit")
def deltas(project: str | None, pipeline: str) -> None:
    """Emit this project's units of work as delta payloads.

    A workflow step and a delta are one unit of work seen from two ends --
    `qmcp/cookbook/delta.py` is the correspondence. This writes the payloads to
    stdout; `dossier deltas ingest` is the other half. Nothing here reaches
    dossier: what crosses is a schema, not an import.
    """
    import importlib
    import json as _json

    from qmcp.addresses import format_address
    from qmcp.cookbook.delta import to_delta
    from qmcp.dashboard import DEFAULT_PROJECT

    owner_repo = project or DEFAULT_PROJECT
    try:
        module = importlib.import_module(f"qmcp.cookbook.{pipeline}")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{pipeline}: no such cookbook pipeline ({exc}). Its steps must be "
            f"importable without a flow runtime -- see qmcp/cookbook/change_impact.py."
        ) from exc

    found = [obj for name, obj in vars(module).items() if name.endswith("_PIPELINE")]
    if not found:
        raise SystemExit(f"{pipeline}: declares no *_PIPELINE to read steps from.")

    owner, _, repo = owner_repo.partition("/")
    payloads = []
    for step in found[0].steps:
        payload = to_delta(step, None, project=owner_repo)
        # The address is what lets dossier name the same row. `to_delta` carries
        # the project and the name; this states the address explicitly so the
        # ingesting side never has to reassemble it.
        payload["links"].append({
            "link_type": "address",
            "target_id": None,
            "target_name": format_address(owner, repo, "delta", step.name),
        })
        payloads.append(payload)

    click.echo(_json.dumps(payloads, indent=2))


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
