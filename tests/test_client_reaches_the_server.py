"""The client's default and the server's default are one number.

**IT COULD NOT REACH ITS OWN SERVER.** `MCPClient()` defaulted to
`http://localhost:3333` while `qmcp.config` served `3141`, so the client this
package ships could not talk to the server this package starts without being
told where it was. Established by running both -- `MCPClient().health()` raised
`ConnectError`, and the same client given `http://127.0.0.1:3141` returned
`{'status': 'healthy'}`.

`qmcp/config.py`'s own comment records the earlier half of that: 3333 is what
the harness served while a control panel looked on 8000. The server moved to
3141 and four call sites did not.

**WHY A TEST AND NOT A CONSTANT.** A shared constant is what this already had --
the literal `3333`, in five places. What it lacked was anything that fails when
one of them moves. These tests do not check the number; they check that the
client's default and the server's settings are the same thing, so `QMCP_PORT`
moves both ends and neither can be changed alone.

THE MUTATION, quoted as it printed. `default_base_url` given the old literal:

    AssertionError: the client's default is not where the server serves
    assert 'http://localhost:3333' == 'http://127.0.0.1:3141'
"""

from __future__ import annotations

import inspect

from qmcp.client.mcp_client import MCPClient, default_base_url
from qmcp.config import get_settings


def test_the_client_default_is_where_the_server_serves() -> None:
    settings = get_settings()

    assert default_base_url() == f"http://{settings.host}:{settings.port}", (
        "the client's default is not where the server serves")


def test_the_constructor_takes_the_derived_default() -> None:
    """`None` rather than a literal in the signature, so it is derived at call
    time and an environment override is not baked in at import."""
    parameter = inspect.signature(MCPClient.__init__).parameters["base_url"]

    assert parameter.default is None
    assert MCPClient().base_url == default_base_url()


def test_an_explicit_base_url_still_wins() -> None:
    """A client pointed at another machine must not be redirected home."""
    assert MCPClient("http://elsewhere:9999").base_url == "http://elsewhere:9999"


def test_the_toolset_default_is_derived_at_construction() -> None:
    """A dataclass default that was a literal is a default_factory now.

    A plain literal would be evaluated at import and could not follow
    `QMCP_PORT`; a factory is called per instance.
    """
    from qmcp.integrations.pydantic_ai.toolsets import QMCPToolset

    assert QMCPToolset().base_url == default_base_url()


def test_no_source_file_still_names_the_port_the_server_left() -> None:
    """The literal is gone from the package, not just from the entry points.

    A scan, because the four call sites were found by grep and not by reading:
    whatever else this proves, it proves nobody has to remember all four.
    """
    from pathlib import Path

    package = Path(__file__).resolve().parent.parent / "qmcp"
    offenders = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if ":3333" in line and "3141" not in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert not offenders, "\n".join(offenders)


def test_the_flow_runner_default_follows_the_same_setting() -> None:
    """A containerised flow reaches the host, on the port the host serves.

    The host part stays `host.docker.internal` -- the caller is in a container
    and the harness is not -- and only the port is derived.
    """
    from qmcp.cli import _default_mcp_url

    assert _default_mcp_url().endswith(f":{get_settings().port}")
    assert "host.docker.internal" in _default_mcp_url()
