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
from qmcp.db.models import InstructionStatus


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
    """Where a containerised flow reaches this machine's harness.

    The host part is `host.docker.internal` because the caller is in a
    container and the harness is not. The port is the one the harness serves,
    read from its settings rather than typed in -- a literal here was 3333
    while the server served 3141.
    """
    from qmcp.config import get_settings

    return os.getenv("MCP_URL",
                     f"http://host.docker.internal:{get_settings().port}")


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
    """Start the MCP server: `uv run qmcp serve`.

    On Windows a running console script holds `Scripts/qmcp.exe`, so a sync
    that reinstalls qmcp -- after pulling a change to its own dependencies --
    fails with os error 32 while this runs. A no-op sync does not touch the
    file. The server is running the code from before that change anyway, so
    the remedy is the restart it needed: stop it, and run the command again.
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
    click.echo("  voice               HITL approval answered by voice; offline, or --live (host)")
    click.echo("  instruct            an instruction taken by voice and recorded, not run; offline")


@cookbook.command("voice")
@click.option("--live", is_flag=True,
              help="ask one question aloud through the configured server and a running engine")
@click.option("--base-url", default=None,
              help="qmcp server URL for --live (default: this machine's configured host:port)")
@click.option("--engine", default="joe", help="which vox.adapters entry to use, for --live")
@click.option("--engine-url", default=None,
              help="where that engine listens, for --live (default: the adapter's own)")
def cookbook_voice(live: bool, base_url: str | None, engine: str, engine_url: str | None) -> None:
    """The voice HITL check: a request queued, answered by voice, read back.

    Offline by default: a qmcp server on an ephemeral port over a database
    made for the run, and vox's deterministic engine in place of a speech
    engine. Scripted answers go through the real path (a yes, an option by
    name, a mismatch, silence, and an open question read back and recorded),
    and each ending is checked. No microphone, speakers or model are needed,
    and the configured queue is not touched.

    `--live` asks one question aloud ("Voice check. Say approve or hold.")
    through the configured server and a running engine, and reports what was
    recorded. It queues one request, `voice-check-<time>`, which expires in
    five minutes.
    """
    from qmcp.integrations.voice.check import run_live, run_offline

    if not live:
        _load_vox(engine, "cookbook voice")
        if not run_offline(echo=click.echo):
            raise SystemExit(1)
        return

    HttpSTT, Pyttsx3TTS, adapter, contract = _load_vox(engine, "cookbook voice --live")
    from qmcp.client import MCPClient

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    resolved_engine = engine_url or getattr(adapter, "DEFAULT_URL", "http://127.0.0.1:8000")
    _voice_preflight(client.base_url, resolved_engine, contract, engine)
    click.echo(f"qmcp:    {client.base_url}")
    click.echo(f"engine:  {resolved_engine} ({engine})")
    with HttpSTT(resolved_engine, contract=contract) as stt:
        if not run_live(client, stt, Pyttsx3TTS(), echo=click.echo):
            raise SystemExit(1)


@cookbook.command("instruct")
def cookbook_instruct() -> None:
    """The spoken-instruction loop, end to end: asked, recorded, consented, run, said back.

    Offline, and only offline: a qmcp server on an ephemeral port over a
    database made for the run, and vox's deterministic engine in place of a
    speech engine. First the inbox: each scripted dialog goes through the real
    path -- the prompt synthesized to a file, every take returned over the
    engine contract, the row recorded over HTTP and read back. Then the loop:
    an instruction recorded, consent asked aloud and answered, the scripted
    runtime run in a directory standing in for the clone, and the summary
    said back, printed as the conversation it was. No microphone, speakers,
    model or agent are needed, nothing is spent, and the configured inbox is
    not touched.
    """
    from qmcp.instructions.check import run_loop, run_offline

    _load_vox("joe", "cookbook instruct")
    inbox = run_offline(echo=click.echo)
    click.echo("")
    loop = run_loop(echo=click.echo)
    if not (inbox and loop):
        raise SystemExit(1)


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


def _voice_preflight(qmcp_url: str, engine_url: str, contract, engine: str) -> None:
    """Refuse to start a voice dialog that cannot finish, saying why.

    Both halves are checked before anything is spoken, so a first run with
    one of them down ends in the command that starts it rather than in a
    connection traceback -- or, worse, a prompt spoken into a room whose
    answer can never be recorded.
    """
    import httpx

    try:
        httpx.get(f"{qmcp_url}/health", timeout=3).raise_for_status()
    except httpx.HTTPError:
        raise SystemExit(
            f"No qmcp server answers at {qmcp_url}. Start one from this clone"
            " with `uv run qmcp serve`, then run this again."
        )
    try:
        httpx.get(contract.url(engine_url, contract.health), timeout=3).raise_for_status()
    except httpx.HTTPError:
        hint = (
            " For joe, in its own checkout: `uv run joe voice setup` once to choose"
            " the microphone, then `uv run joe backend`."
            if engine == "joe"
            else ""
        )
        raise SystemExit(
            f"No speech engine answers at {engine_url} ({engine}).{hint}"
            " Then run this again."
        )


def _vox_shadowed_by() -> str | None:
    """The directory `import vox` wrongly resolves to, or None.

    A directory named `vox` without an `__init__.py` on any `sys.path` root
    makes `vox` an empty namespace package that hides the installed one:
    its submodules still import, so a check on `vox.stt` passes, while
    `from vox import HttpSTT` fails. qmcp's editable install puts the
    project root on `sys.path`, which is why the submodule moved from a
    top-level `vox/` to `vendor/vox`; a clone updated across that move can
    keep the old directory.
    """
    try:
        import vox
    except ImportError:
        return None
    if getattr(vox, "__file__", None) is not None:
        return None
    paths = list(getattr(vox, "__path__", []))
    return paths[0] if paths else "an unknown location"


def _qmcp_already_serving(host: str, port: int) -> str | None:
    """The version already answering /health here as qmcp, or None.

    Checked before binding, because uvicorn's own bind failure arrives after
    the startup banner and reads as a broken server rather than as the fact
    that a working one is already up.
    """
    import httpx

    try:
        response = httpx.get(f"http://{host}:{port}/health", timeout=2)
        body = response.json()
    except Exception:
        return None
    if response.status_code == 200 and "status" in body:
        return str(body.get("version", "unknown"))
    return None


def _run_server(host: str | None, port: int | None, reload: bool) -> None:
    settings = get_settings()

    actual_host = host or settings.host
    actual_port = port or settings.port

    serving = _qmcp_already_serving(actual_host, actual_port)
    if serving is not None:
        raise SystemExit(
            f"qmcp {serving} already serves http://{actual_host}:{actual_port},"
            " and every CLI command talks to it over HTTP -- a second server"
            " is not needed. To run another beside it, pass a different"
            " --port."
        )

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
              help="include requests that have been answered or have expired")
def human_list(database: Path | None, show_all: bool) -> None:
    """What is waiting on a person, oldest first."""
    from datetime import UTC, datetime

    from sqlmodel import Session, create_engine, select

    from qmcp.db.models import HumanRequest, HumanResponse

    # SQLite stores naive datetimes, treat as UTC
    now = datetime.now(UTC).replace(tzinfo=None)
    engine = create_engine(f"sqlite:///{Path(database or _configured_database()).as_posix()}")
    with Session(engine) as session:
        requests = session.exec(
            select(HumanRequest).order_by(HumanRequest.created_at)).all()
        answers = {r.request_id: r for r in session.exec(select(HumanResponse)).all()}

        shown = 0
        for request in requests:
            reply = answers.get(request.id)
            expires = request.expires_at
            if expires is not None and expires.tzinfo is not None:
                expires = expires.astimezone(UTC).replace(tzinfo=None)
            expired = reply is None and expires is not None and expires <= now
            if (reply is not None or expired) and not show_all:
                continue
            shown += 1
            mark = "[x]" if expired else "[?]" if reply is None else "[=]"
            click.echo(f"  {mark} {request.id}")
            click.echo(f"      {request.prompt}")
            if request.options:
                click.echo(f"      options: {', '.join(request.options)}")
            if reply is not None:
                click.echo(f"      answered: {reply.response}"
                           + (f"  ({reply.responded_by})" if reply.responded_by else ""))
            elif expired:
                click.echo(f"      expired: {expires:%Y-%m-%d %H:%M} UTC, unanswered")
            click.echo("")

        if not shown:
            click.echo("  Nothing is waiting on a person."
                       + ("" if show_all else "  (--all includes answered and expired ones.)"))
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


@human.command("voice")
@click.argument("request_id", required=False)
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
@click.option("--engine", default="joe",
              help="which vox.adapters entry to use for speech recognition")
@click.option("--engine-url", default=None,
              help="where that engine listens (default: the adapter's own)")
@click.option("--duration", default=5.0, type=float, help="seconds to listen per attempt")
@click.option("--max-retries", default=2, type=int,
              help="re-asks before giving up on an unclear spoken answer")
@click.option("--forever", is_flag=True,
              help="keep answering requests as they arrive, instead of just one")
@click.option("--poll-interval", default=2.0, type=float,
              help="seconds between checks for the next pending request, with --forever")
def human_voice(request_id: str | None, base_url: str | None, engine: str,
                engine_url: str | None, duration: float, max_retries: int,
                forever: bool, poll_interval: float) -> None:
    """Answer one (or, with --forever, every) pending request by voice.

    Speaks the prompt through a synthesizer, listens via a running speech
    engine, and submits the answer: one of the request's options, or for a
    request with none, the transcript read back and confirmed. Unlike
    `human list` and `human respond`, this goes over HTTP rather than straight
    to the database: recognition only exists behind a running `qmcp serve` and
    a running engine, so there is no offline path here to preserve.

    --engine names a module in `vox.adapters`; vox states the contract and
    names no engine of its own.

    REQUEST_ID is optional: without it, the oldest pending request is
    answered. vox and its synthesizer install with the default dependencies:
    `git submodule update --init vendor/vox`, then `uv sync`.
    """
    HttpSTT, Pyttsx3TTS, adapter, contract = _load_vox(engine, "human voice")

    from qmcp.client import HumanRequestExpiredError, MCPClient, MCPClientError
    from qmcp.integrations.voice import UnclearResponse, VoiceApprovalLoop

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    resolved_engine = engine_url or getattr(adapter, "DEFAULT_URL", "http://127.0.0.1:8000")
    _voice_preflight(client.base_url, resolved_engine, contract, engine)
    loop = VoiceApprovalLoop(
        stt=HttpSTT(resolved_engine, contract=contract),
        tts=Pyttsx3TTS(),
        client=client,
        max_retries=max_retries,
        listen_duration=duration,
    )
    if forever:
        click.echo("Listening for pending requests by voice. Ctrl+C to stop.")
        try:
            answered = loop.run_forever(poll_interval=poll_interval)
        except KeyboardInterrupt:
            click.echo("\nStopped.")
        else:
            click.echo(f"Answered {answered} request(s).")
        if loop.unanswered:
            click.echo(
                "Asked once, no usable answer, still pending: "
                + ", ".join(loop.unanswered)
                + ". `qmcp human voice <id>` asks one again."
            )
        return

    if request_id is None:
        pending = client.list_human_requests(status_filter="pending", limit=1, oldest_first=True)
        if not pending:
            click.echo("Nothing is waiting on a person.")
            return
        request_id = pending[0].id

    try:
        response = loop.run_once(request_id)
    except UnclearResponse as exc:
        raise SystemExit(str(exc))
    except HumanRequestExpiredError as exc:
        raise SystemExit(str(exc))
    except MCPClientError as exc:
        raise SystemExit(str(exc))

    click.echo(f"  {request_id} answered {response.response!r} (by voice).")


def _load_vox(engine: str, command: str):
    """vox's client and synthesizer, and the named engine's adapter and contract.

    Exits with the remedy when vox cannot be imported, naming a shadowing
    directory where one is the cause.
    """
    try:
        import importlib

        from vox import HttpSTT
        from vox.adapters.pyttsx3 import Pyttsx3TTS
    except ImportError:
        shadow = _vox_shadowed_by()
        if shadow:
            raise SystemExit(
                "vox is shadowed: `import vox` resolves to the directory"
                f" {shadow}, which is not the package, so vox's own names are"
                " missing. That is a leftover top-level `vox/` directory -- the"
                " submodule lives at vendor/vox. Delete the stray directory and"
                " run this again."
            )
        raise SystemExit(
            "vox is not importable. It installs with the default dependencies:"
            " `git submodule update --init vendor/vox`, then `uv sync`, then"
            f" `uv run qmcp {command}` again. If the sync fails with os error 32,"
            " a server started with `uv run qmcp serve` holds `qmcp.exe`: stop it"
            " first."
        )

    try:
        adapter = importlib.import_module(f"vox.adapters.{engine}")
        contract = getattr(adapter, engine.upper())
    except (ImportError, AttributeError):
        raise SystemExit(f"No vox engine adapter named {engine!r}.")
    return HttpSTT, Pyttsx3TTS, adapter, contract


# One mark per kind of state: recorded, unresolved, on the way through consent
# or a run, ended without running, and ended with a run that succeeded.
_INSTRUCTION_MARKS = {
    "recorded": "[=]", "unresolved": "[?]",
    "asking": "[>]", "consented": "[>]", "acting": "[>]",
    "refused": "[x]", "unanswered": "[x]", "failed": "[x]",
    "done": "[+]",
}


def _print_instruction(row: dict) -> None:
    """One row, as `instruct`, `instructions list` and `instructions show` print it."""
    status = row.get("status")
    mark = _INSTRUCTION_MARKS.get(status, "[?]")
    project = row.get("project") or "unresolved"
    when = (row.get("created_at") or "")[:16].replace("T", " ")
    click.echo(f"  {mark} {row['id']}  {project}  {row.get('source')}  {when}"
               + (f"  {status}" if status not in ("recorded", "unresolved") else ""))
    click.echo(f"      {row['text']}")
    detail = row.get("detail") or {}
    if status == "unresolved":
        candidates = detail.get("candidates") or []
        click.echo("      candidates: " + (", ".join(candidates) if candidates else "none")
                   + f"  ({detail.get('rule')})")
    if row.get("runtime"):
        click.echo(f"      runtime: {row['runtime']}  clone: {row.get('cwd')}"
                   + (f"  exit {row['exit_code']}" if row.get("exit_code") is not None else ""))


@cli.command("instruct")
@click.argument("text", required=False)
@click.option("--project", default=None,
              help="the project it is for, stated outright; skips the matching")
@click.option("--source", type=click.Choice(["typed", "page"]), default="typed",
              show_default=True, help="how a typed instruction arrived")
@click.option("--voice", is_flag=True, help="speak the instruction instead of typing it")
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
@click.option("--engine", default="joe",
              help="which vox.adapters entry to use for speech recognition, with --voice")
@click.option("--engine-url", default=None,
              help="where that engine listens, with --voice (default: the adapter's own)")
@click.option("--duration", default=30.0, type=float, show_default=True,
              help="seconds an instruction may take: the cap, with --voice")
@click.option("--pause-ms", default=1500, type=int, show_default=True,
              help="how long a pause ends the take, with --voice")
@click.option("--max-retries", default=2, type=int, show_default=True,
              help="re-asks before giving up on an instruction nobody confirmed")
def instruct(text: str | None, project: str | None, source: str, voice: bool,
             base_url: str | None, engine: str, engine_url: str | None,
             duration: float, pause_ms: int, max_retries: int) -> None:
    """Record an instruction against a project. Recording executes nothing.

    Typed, TEXT is recorded over HTTP and the row is printed: which project
    it resolved to, by whole-word match against the roster in governance/qm,
    or `unresolved` with the candidates when none or several matched.
    --project states it outright and skips the matching.

    --voice asks "What should be done?" aloud, listens with a long cap and a
    long pause (an instruction has pauses mid-thought), reads the transcript
    back ("I heard: ... Say record or again."), and records on a yes. An
    ambiguous project is asked back as a closed choice by name; a missing
    one is asked for once. The engine must be reachable, as for
    `qmcp human voice`.
    """
    from qmcp.client import MCPClient

    if voice and text is not None:
        raise click.UsageError("TEXT and --voice are two ways to give one instruction; pass one.")
    if not voice and text is None:
        raise click.UsageError("Pass the instruction as TEXT, or --voice to speak it.")

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    if not voice:
        _print_instruction(client.create_instruction(text, source=source, project=project))
        return

    HttpSTT, Pyttsx3TTS, adapter, contract = _load_vox(engine, "instruct --voice")
    from qmcp.instructions import roster_names
    from qmcp.instructions.dialog import InstructionDialog
    from qmcp.integrations.voice import UnclearResponse

    resolved_engine = engine_url or getattr(adapter, "DEFAULT_URL", "http://127.0.0.1:8000")
    _voice_preflight(client.base_url, resolved_engine, contract, engine)
    with HttpSTT(resolved_engine, contract=contract) as stt:
        dialog = InstructionDialog(stt=stt, tts=Pyttsx3TTS(), client=client,
                                   names=roster_names(), max_retries=max_retries,
                                   listen_duration=duration, pause_ms=pause_ms)
        try:
            row = dialog.run_once()
        except UnclearResponse as exc:
            raise SystemExit(str(exc))
    _print_instruction(row)


@cli.group("instructions")
def instructions() -> None:
    """The instruction inbox: what people have asked for, and for which project.

    Over HTTP, like `instruct`: the inbox is the server's record.
    """


@instructions.command("list")
@click.option("--status", "status_filter",
              # Read from the model, so the choice cannot name a state the row cannot hold.
              type=click.Choice([s.value for s in InstructionStatus]),
              default=None, help="only rows in this state")
@click.option("--limit", default=50, type=int, show_default=True)
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
def instructions_list(status_filter: str | None, limit: int, base_url: str | None) -> None:
    """What has been recorded, newest first, and where each row is: `recorded`
    and `unresolved` have had nothing asked or run; the rest are on the path
    `instructions act` walks, and only `acting` is a run in progress."""
    from qmcp.client import MCPClient

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    rows = client.list_instructions(status=status_filter, limit=limit)
    if not rows:
        click.echo("  Nothing recorded" + (f" as {status_filter}." if status_filter else "."))
        return
    for row in rows:
        _print_instruction(row)
    click.echo(f"  {len(rows)} instruction(s).")


@instructions.command("show")
@click.argument("instruction_id")
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
def instructions_show(instruction_id: str, base_url: str | None) -> None:
    """One instruction, with the evidence for its project."""
    import json as _json

    from qmcp.client import MCPClient, MCPClientError

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    try:
        row = client.get_instruction(instruction_id)
    except MCPClientError as exc:
        raise SystemExit(str(exc))
    _print_instruction(row)
    click.echo("      detail: " + _json.dumps(row.get("detail") or {}, sort_keys=True))


@instructions.command("act")
@click.argument("instruction_id")
@click.option("--runtime", "runtime_name", default=None, envvar="QMCP_AGENT_RUNTIME",
              help="which agent carries it out, by the name its adapter declares;"
                   " required, or set QMCP_AGENT_RUNTIME. There is no default.")
@click.option("--budget", default=0, type=int, show_default=True,
              help="runs this command may make; 0 declares what would be asked and stops")
@click.option("--cwd", type=click.Path(path_type=Path), default=None,
              help="the clone to run in; without it, the one the project's last act ran in,"
                   " from qmcp's record")
@click.option("--voice", is_flag=True,
              help="answer the consent by voice here, as `qmcp human voice` would")
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
@click.option("--engine", default="joe",
              help="which vox.adapters entry to use for speech recognition, with --voice")
@click.option("--engine-url", default=None,
              help="where that engine listens, with --voice (default: the adapter's own)")
@click.option("--poll-interval", default=1.0, type=float, show_default=True,
              help="seconds between reads of the pending listing while the consent waits")
def instructions_act(instruction_id: str, runtime_name: str | None, budget: int,
                     cwd: Path | None, voice: bool, base_url: str | None, engine: str,
                     engine_url: str | None, poll_interval: float) -> None:
    """Act on one instruction: declare the spend, ask consent, run only on approve.

    The clone is --cwd when it is given; without it, the clone the project's
    last act ran in, from qmcp's record. The runtime is handed the project's
    earlier instructions and outcomes from the same record: continuity comes
    from qmcp, not the model. A consent request
    `instruction-<id>` with the options approve and hold goes on the human
    queue, saying the instruction, the project, the clone, the runtime and
    the budget, and expires after a fixed wait
    (`qmcp.instructions.act.CONSENT_SECONDS`). Approve runs
    the runtime in the clone and records the outcome as `done` or `failed`;
    hold records `refused`; silence records `unanswered`. Nothing runs on any
    path but approve, and the declaration is recorded on every path.

    --budget is the number of runs this command may make, and 0 is the
    default: it resolves the clone, declares what would be asked, and stops.
    --runtime has no default. `scripted` runs nothing and is for checks.
    """
    from qmcp.client import MCPClient
    from qmcp.instructions.act import NoSuchInstruction, act
    from qmcp.integrations.agents import runtime_named, runtime_names
    from qmcp.spend import Budget, render

    if not runtime_name:
        raise click.UsageError(
            "--runtime is required and has no default: a default that ran an agent would"
            " be a paid call nobody chose. Pass one of "
            f"{', '.join(runtime_names())}, or set QMCP_AGENT_RUNTIME.")
    try:
        runtime = runtime_named(runtime_name)
    except KeyError as exc:
        raise click.UsageError(str(exc.args[0]))

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    stt = tts = None
    if voice:
        HttpSTT, Pyttsx3TTS, adapter, contract = _load_vox(engine, "instructions act --voice")
        resolved_engine = engine_url or getattr(adapter, "DEFAULT_URL", "http://127.0.0.1:8000")
        _voice_preflight(client.base_url, resolved_engine, contract, engine)
        stt, tts = HttpSTT(resolved_engine, contract=contract), Pyttsx3TTS()

    from qmcp.instructions.spoken import say, starting, summarise

    def show(state: str, text: str) -> None:
        click.echo(f"  [{state}] {text}")
        # Between the approval and the run, so the person who said approve
        # hears that it took rather than waiting on silence.
        if state == "acting" and tts is not None:
            say(starting(_instruction_row(instruction_id).get("project")), tts, stt)

    try:
        try:
            done = act(instruction_id, runtime, Budget(authorised=budget), client=client,
                       cwd=cwd, stt=stt, tts=tts, on_event=show, poll_interval=poll_interval)
        except NoSuchInstruction as exc:
            raise SystemExit(f"{exc}. `qmcp instructions list` shows the inbox.")

        click.echo(f"  {done.instruction_id}  {done.status}  stages: {' > '.join(done.stages)}")
        if done.cwd:
            click.echo(f"      clone: {done.cwd}"
                       + (f"  carrying {len(done.carried)} earlier instruction(s)"
                          if done.carried else ""))
        if done.request_id:
            click.echo(f"      consent: {done.request_id}  answered: {done.answer!r}")
        if done.why:
            click.echo(f"      {done.why}")
        if done.outcome is not None:
            click.echo(f"      exit {done.outcome.exit_code} after"
                       f" {done.outcome.elapsed_seconds:.1f}s; spent {done.outcome.spent}")
            click.echo("      " + (done.outcome.text.strip().splitlines() or [""])[0][:200])
        click.echo(render(done.declared))
        # The row, not `done`, is what is said: the same row says the same
        # thing here, in `instructions say`, and in the offline check.
        summary = summarise(_instruction_row(done.instruction_id), why=done.why)
        click.echo(f"  {'said' if tts is not None else 'summary'}: {summary}")
        if tts is not None:
            say(summary, tts, stt)
    finally:
        if stt is not None and hasattr(stt, "close"):
            stt.close()
    if done.status == "failed":
        raise SystemExit(1)


def _instruction_row(instruction_id: str) -> dict:
    """One inbox row as the server would serve it, read from the database the
    act wrote, or an empty mapping when there is none."""
    from qmcp.db.models import Instruction
    from qmcp.instructions import act as act_module

    with act_module.configured_rows()() as session:
        row = session.get(Instruction, instruction_id)
        return row.model_dump(mode="json") if row is not None else {}


@instructions.command("say")
@click.argument("instruction_id")
@click.option("--speak", is_flag=True,
              help="say it aloud through vox's local synthesizer as well; no engine is contacted")
@click.option("--base-url", default=None,
              help="qmcp server URL (default: this machine's configured host:port)")
def instructions_say(instruction_id: str, speak: bool, base_url: str | None) -> None:
    """What an instruction came to, in the few sentences `act --voice` says.

    Printed, and with --speak said aloud as well. The whole outcome is
    `qmcp instructions show <id>`.
    """
    from qmcp.client import MCPClient, MCPClientError
    from qmcp.instructions.spoken import say, summarise

    client = MCPClient(base_url=base_url) if base_url else MCPClient()
    try:
        row = client.get_instruction(instruction_id)
    except MCPClientError as exc:
        raise SystemExit(str(exc))
    summary = summarise(row)
    click.echo(summary)
    if speak:
        _, Pyttsx3TTS, _, _ = _load_vox("joe", "instructions say --speak")
        say(summary, Pyttsx3TTS())


@cli.group("threads")
def threads() -> None:
    """Conversations with an assistant, as units of work.

    Reads a local export. Nothing here calls a paid service or needs a
    credential -- `governance/qm/records/DRAFT-no-unattended-spending.md` is
    why, and an API source will be a second source behind the same contract
    rather than a change to this one.
    """


def _sources(root, sessions=None):
    """Every thread source, each pointed at the store it actually reads.

    Two roots, not one. The web exports live in a cache this project unpacks
    into; Claude Code sessions live in a store somebody else owns and this only
    reads. Passing one root to both would send the session reader looking in an
    export cache and report zero sessions, which reads like an empty machine.
    """
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads
    from qmcp.threads.claudecode import SESSION_ROOT, ClaudeCodeThreads

    return [
        ClaudeThreads(root=root),
        ChatGPTThreads(root=root),
        ClaudeCodeThreads(root=Path(sessions) if sessions else SESSION_ROOT),
    ]


def _root(root):
    from qmcp.threads.cache import DEFAULT_ROOT

    return Path(root) if root else DEFAULT_ROOT


@threads.command("sources")
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="the export cache to read instead of the default")
@click.option("--sessions", type=click.Path(path_type=Path), default=None,
              help="the Claude Code session store to read instead of the default")
def threads_sources(root: Path | None, sessions: Path | None) -> None:
    """What is cached, per assistant, and what pulling it would cost."""
    for source in _sources(_root(root), sessions):
        click.echo(source.describe())
        click.echo("")


@threads.command("import")
@click.argument("export", type=click.Path(exists=True, path_type=Path))
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="the cache to unpack into instead of the default")
@click.option("--source", type=click.Choice(["claude", "chatgpt"]), default=None,
              help="which service wrote it, when the shape cannot say")
@click.option("--dry-run", is_flag=True, help="report and write nothing")
def threads_import(export: Path, root: Path | None, source: str | None,
                   dry_run: bool) -> None:
    """Unpack an official data export into the cache.

    EXPORT is the ZIP the service produced, or a `conversations.json` already
    unpacked from one.

    
    THE API IS NOT A ROUTE TO THIS DATA. Neither Anthropic's nor OpenAI's API
    exposes the conversation history of claude.ai or chatgpt.com -- they are
    products for making new model calls, against different storage, with no
    endpoint that lists your threads. The export is the sanctioned route and
    requesting it is the account holder's, in the web interface.

    Nothing here reaches the network or spends anything.
    """
    from qmcp.threads.importer import positional, render, unpack

    directory = _root(root)
    try:
        report = unpack(export, directory, source=source, dry_run=dry_run)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"{export.name}: {exc}") from exc

    click.echo(render(report, directory, dry_run))

    fallback = list(positional(report))
    if fallback:
        click.echo("")
        click.echo(f"  {len(fallback)} conversation(s) carried no id and were "
                   f"named by position.")
        click.echo("  A later export with one deleted shifts those names, and "
                   "the index would")
        click.echo("  read that as many threads diverging at once. Worth "
                   "knowing before it does.")

    if not dry_run and report.total:
        click.echo("")
        click.echo("  Next: uv run qmcp threads index --write")


@threads.command("index")
@click.option("--root", type=click.Path(path_type=Path), default=None)
@click.option("--sessions", type=click.Path(path_type=Path), default=None,
              help="the Claude Code session store to read instead of the default")
@click.option("--write", is_flag=True, help="write the index")
@click.option("--check", "check", is_flag=True,
              help="re-derive from the files and report drift; writes nothing")
def threads_index(root: Path | None, sessions: Path | None, write: bool,
                  check: bool) -> None:
    """Index the cache, keeping what earlier indexes knew.

    Nothing is overwritten. A conversation somebody kept talking in produces a
    later version of the same strand, and whether it *grew* or *diverged* is
    recorded rather than resolved -- an export that disagrees with an earlier
    record of itself is a finding, and the prior digest is the only evidence.
    """
    import json as _json

    from qmcp.threads import index as index_module

    directory = _root(root)
    path = directory / index_module.INDEX_NAME
    sources = _sources(directory, sessions)

    entries = index_module.build(sources)
    unreadable = {
        source.name: [item.path for item in source.unreadable]
        for source in sources if getattr(source, "unreadable", None)
    }

    if check:
        if not path.is_file():
            raise SystemExit(
                f"no index at {path}. `--check` compares a written index "
                f"against the files; there is nothing to compare."
            )
        found = _json.loads(path.read_text(encoding="utf-8"))
        problems = index_module.drift(found, entries)
        if problems:
            click.echo(f"{len(problems)} disagreement(s) between the index and "
                       f"the files:")
            for problem in problems:
                click.echo(f"  {problem}")
            raise SystemExit(1)
        click.echo("The index still describes the files it was built from.")
        click.echo("Only the cache layer is compared. The archive layer is "
                   "history and cannot be re-derived from the files, which is "
                   "what makes it worth keeping.")
        return

    merged, changed = index_module.merge(index_module.load(path), entries)
    document = index_module.document(merged, unreadable)

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(document, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
        click.echo(index_module.render(document, changed))
        click.echo("")
        click.echo(f"Written to {path}. It describes this machine's cache and "
                   f"is not committed anywhere.")
        return

    click.echo(index_module.render(document, changed))
    click.echo("")
    click.echo("Nothing was written. Pass --write to keep it.")


@threads.command("dashboard")
def threads_dashboard() -> None:
    """Where the archive is read. It is not here.

    
    THE ARCHIVE HAS ONE HUMAN SURFACE, AND IT IS THE CONTROL PANEL. This
    project rendered a second one -- a self-contained HTML page -- and it was
    removed rather than kept: two views of one dataset are two definitions of
    what a figure means, and they drift the first time one is fixed.

    The commands beside this one stay, because a command line is for machines
    and for debugging. `governance/qm/PRINCIPLES.md` P13.
    """
    click.echo("  The archive is read in the control panel:")
    click.echo("")
    click.echo("    uv run qmcp serve                  # here, on loopback")
    click.echo("    uv run dossier dashboard           # there, the Threads tab")
    click.echo("")
    click.echo("  It reads this harness over HTTP and imports nothing from it.")
    click.echo("  For debugging, `qmcp threads list` and `--check` are still here.")


@threads.command("consolidate")
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="the export cache to read instead of the default")
@click.option("--sessions", type=click.Path(path_type=Path), default=None,
              help="the Claude Code session store to read instead of the default")
@click.option("--corpus", type=click.Path(path_type=Path),
              default=Path("governance") / "qm",
              help="the corpus whose workspace names the projects")
@click.option("--by-project", is_flag=True,
              help="also list which threads each project has")
def threads_consolidate(root: Path | None, sessions: Path | None,
                        corpus: Path, by_project: bool) -> None:
    """Which projects the archive is about, read against the workspace roster.

    Reads files already on this disk and spends nothing. The count of threads
    about no project is reported first and never omitted -- a consolidator that
    printed only its hits would read as though it had placed everything.
    """
    from qmcp.spend import Budget
    from qmcp.threads import consolidate as consolidator

    if not (corpus / "ci" / "workspace.yaml").is_file():
        raise click.ClickException(
            f"no workspace at {corpus.as_posix()}/ci/workspace.yaml. The "
            f"roster names the projects, so without it every thread would be "
            f"about none of them, which is a different answer.")

    names = consolidator.roster(corpus)
    collected: list = []
    for source in _sources(_root(root), sessions):
        collected.extend(source.fetch([], Budget(authorised=0)))

    reading = consolidator.consolidate(collected, names)
    click.echo(reading.summary())
    if by_project:
        click.echo("")
        for project, found in sorted(reading.by_project().items()):
            click.echo(f"  {project}: {len(found)} thread(s)")


@threads.command("list")
@click.option("--root", type=click.Path(path_type=Path), default=None)
@click.option("--diverged", is_flag=True,
              help="only threads whose export disagrees with an earlier one")
def threads_list(root: Path | None, diverged: bool) -> None:
    """What the index holds. Reads the index, not the files."""
    import json as _json

    from qmcp.threads import index as index_module

    path = _root(root) / index_module.INDEX_NAME
    if not path.is_file():
        raise SystemExit(
            f"no index at {path}. `uv run qmcp threads index --write` builds "
            f"one; nothing here reads the files, so an absent index is an "
            f"absent answer rather than an empty one."
        )

    found = _json.loads(path.read_text(encoding="utf-8"))
    rows = found.get("threads") or []
    if diverged:
        rows = [row for row in rows
                if any(c["kind"] == "diverged" for c in row["history"])]

    if not rows:
        click.echo("No thread matches." if diverged else "The index is empty.")
        return

    for row in rows:
        mark = "[!]" if any(c["kind"] == "diverged" for c in row["history"]) else "   "
        click.echo(f"  {mark} {row['source']}/{row['id']}  "
                   f"{row['turns']} turn(s)  {row['digest']}")
        if row.get("title"):
            click.echo(f"       {row['title']}")
    click.echo("")
    click.echo(f"{len(rows)} thread(s), from an index generated "
               f"{found['generated_at']}. An export is a snapshot.")


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

    repo = _package_repo_root()
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


# The seed runner is reached through the governance submodule rather than copied
# into this repository, so a change to how a workflow is simulated lands there
# once and every fork picks it up on its next pin.
_PREFLIGHT_RUNNER = Path("governance", "qm", "project-seed", "ci", "run_workflows_locally.py")


def _package_repo_root() -> Path:
    """The checkout this package was imported from, independent of the cwd.

    `_find_repo_root` walks up from the working directory, which is right for a
    command run inside a project and wrong for one that must act on this
    repository whatever directory it was started in.
    """
    return Path(__file__).resolve().parent.parent


# `help_option_names=[]` removes click's own `--help` from this command, so it
# reaches the runner like every other argument: the runner's options are this
# command's options, and its help page is the one that describes them.
@cli.command("preflight", context_settings={
    "ignore_unknown_options": True,
    "help_option_names": [],
})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def preflight(ctx: click.Context, args: tuple[str, ...]) -> None:
    """Run this repository's workflows locally, as the runner would.

    A thin route to `governance/qm/project-seed/ci/run_workflows_locally.py`:
    every argument is passed through unchanged (`--event`, `--ref`,
    `--base-ref`, `--head-ref`, `--workflows`, and `--help`, which is the
    runner's), the script runs under this interpreter from the repository
    root, and its exit status is this command's. The first `--` is the
    argument separator and is consumed before the runner sees it, wherever it
    stands. Nothing is decided here, and the only line printed here is the one
    saying the runner is not in the tree.

        uv run qmcp preflight                              # a PR into main
        uv run qmcp preflight --event push --ref main
        uv run qmcp preflight --base-ref origin/main

    The script's own docstring lists what it cannot reproduce -- `uses:` steps
    and the runner image above all -- which is why a pass is evidence and not
    proof.
    """
    repo = _package_repo_root()
    script = repo / _PREFLIGHT_RUNNER
    if not script.is_file():
        # `is_file`, not `exists`: a directory at the runner's path would pass
        # an existence check and leave the interpreter to report that it found
        # no module there. The check cannot tell an unchecked-out submodule
        # from one pinned before the runner existed; the message names both,
        # and the command it gives is right for the first and harmless for the
        # second.
        click.echo(
            f"the runner is not at {_PREFLIGHT_RUNNER.as_posix()}: the governance "
            "submodule is not checked out, or is pinned before the runner "
            "existed. Run `git submodule update --init governance/qm`.",
            err=True,
        )
        ctx.exit(2)
    # Run from the repository root rather than the cwd: the runner's
    # `--workflows` default is a relative path, and the repository is the
    # thing being checked whatever directory the command was started in.
    result = subprocess.run([sys.executable, str(script), *args], cwd=repo)
    ctx.exit(result.returncode)


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


# =============================================================================
# The routes three modules already told people to run
#
# `qmcp.topology_view`, `qmcp.orchestration` and `qmcp.localmodel` each open
# with a `uv run qmcp ...` line, and until this section none of those commands
# existed. `tests/test_declared_commands.py` is what keeps that from happening
# again, and it names the three that are still only claimed.
# =============================================================================


@cli.group("topology")
def topology() -> None:
    """The shapes this harness knows, and the one it runs a model through.

    A window rather than a description. `qmcp.topology_view` holds boxes and
    arrows with no coordinates, no glyphs and no colours; what is printed here
    is one rendering of that, and the browser front end draws the same payload
    differently without either being more correct.
    """


def _render_view(view: object) -> str:
    """One view as text. A terminal drops what it has no room for.

    Notes are shown and weights are not: a weight needs a length to mean
    anything, and a column of numbers beside a list of boxes would be reporting
    a measurement in a form nobody can read it in.
    """
    lines = [f"{view.topology} (level {view.level}) -- {view.caption}"]
    marks = list(view.marks)
    if view.is_refused:
        marks.append("refused here")
    if marks:
        lines.append(f"  declares: {', '.join(marks)}")
    lines.append("")
    for box in view.boxes:
        count = f" x{box.count}" if box.count is not None else ""
        note = f"  -- {box.note}" if box.note else ""
        lines.append(f"  [{box.kind:<6}] {box.label}{count}{note}")
    if view.arrows:
        lines.append("")
        for arrow in view.arrows:
            label = f" ({arrow.label})" if arrow.label else ""
            mark = "" if arrow.kind == "flow" else f" [{arrow.kind}]"
            lines.append(f"  {arrow.frm} -> {arrow.to}{label}{mark}")
    return "\n".join(lines)


@topology.command("gallery")
@click.option("--level", type=click.IntRange(0, 2), default=0,
              help="0 black box, 1 the parts, 2 the flow between them")
def topology_gallery(level: int) -> None:
    """Every topology this harness knows, at one level."""
    from qmcp import governed
    from qmcp import topology_view as tv

    for view in [*tv.gallery(level=level), governed.view(level=level)]:
        click.echo(_render_view(view))
        click.echo("")


@topology.command("show")
@click.argument("kind")
@click.option("--level", type=click.IntRange(0, 2), default=2,
              help="0 black box, 1 the parts, 2 the flow between them")
def topology_show(kind: str, level: int) -> None:
    """One topology at one resolution.

    KIND is a name from `topology gallery`, or `governed` for the seam a model
    is called through.
    """
    from qmcp import governed
    from qmcp import topology_view as tv
    from qmcp.agentframework.models.enums import TopologyType

    if kind == "governed":
        click.echo(_render_view(governed.view(level=level)))
        return
    try:
        wanted = TopologyType(kind)
    except ValueError:
        raise click.ClickException(
            f"no topology named {kind!r}. `qmcp topology gallery` lists them.")
    click.echo(_render_view(tv.view_of(wanted, level=level)))


@cli.group("orchestration")
def orchestration() -> None:
    """What each topology would do, declared before anything runs one."""


@orchestration.command("plane")
def orchestration_plane() -> None:
    """Every topology's status and what running it would do.

    Reports the declaration rather than establishing it. A plane that worked
    these out by running something would have already done the thing it was
    deciding about.
    """
    from qmcp import orchestration as plane

    click.echo(plane.render())
    drift = plane.undeclared()
    if drift:
        click.echo("")
        click.echo("Attested acts this module restates that the corpus no "
                   "longer names, or vice versa:")
        for line in drift:
            click.echo(f"  - {line}")


@cli.group("localmodel")
def localmodel() -> None:
    """Standing a local model up, and being able to do it again.

    **A deployment decision, not governance.** This is the one place a vendor
    is allowed to appear, because it is this project's own operational tooling
    rather than a rule anybody adopts by reference.
    """


@localmodel.command("check")
def localmodel_check() -> None:
    """What is on this machine, before anything is installed. Installs nothing."""
    from qmcp.localmodel import look

    check = look()
    click.echo(f"  installer:  {check.installer or 'none found'}")
    click.echo(f"  ollama:     {check.ollama or 'not installed'}")
    click.echo(f"  models dir: {check.models_dir or 'not set'}")
    click.echo(f"  gpu:        {check.gpu or 'none reported'}"
               + (f" ({check.vram_mb} MB)" if check.vram_mb else ""))
    for volume in check.volumes:
        click.echo(f"  volume:     {volume.human()}")
    best = check.best_volume
    click.echo(f"  roomiest:   {best.name if best else 'nothing roomy enough'}")
    if check.system_drive_is_tight:
        click.echo("  the system drive is tight; --models-dir elsewhere")
    for blocker in check.blockers:
        click.echo(f"  blocked by: {blocker}")


@localmodel.command("plan")
@click.option("--models-dir", default=None,
              help="where models should be kept, if not the default")
def localmodel_plan(models_dir: str | None) -> None:
    """The exact commands, printed rather than run.

    A step that downloads five gigabytes should be something a person read
    before it started, and something they can run again later without this
    module being involved.
    """
    from qmcp.localmodel import look, plan

    click.echo(plan(look(), models_dir=models_dir).render())


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
