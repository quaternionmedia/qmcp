"""CLI interface for QMCP.

Provides commands for:
- Starting the MCP server
- Listing registered tools
- Development utilities
"""

import click
import uvicorn

from qmcp import __version__
from qmcp.config import get_settings


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
