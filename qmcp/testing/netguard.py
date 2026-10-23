"""No connection leaves this machine: every attempt is refused, and remembered.

`install()` replaces `socket.getaddrinfo`, `socket.socket.connect` and
`socket.socket.connect_ex` in this process. A lookup or a connection whose
host is not loopback is refused the way an offline machine refuses it -- a
lookup with `socket.gaierror`, a connection with `ConnectionRefusedError` --
and recorded, so code that catches the refusal and carries on is still caught.
`taken()` returns what was recorded and forgets it. Inside `permitted()`
nothing is refused or recorded: a test that is about a real service says so.

A Python child process is guarded too, when it starts with the environment
`child_environment()` returns: its `PYTHONPATH` leads with a directory whose
`sitecustomize.py` installs this module before the child's own code runs, and
its attempts are appended to the file `NETGUARD_LOG` names. `UV_OFFLINE` is
set, so a `uv` the child runs uses what is installed and fetches nothing.

What this cannot see: a child that is not Python, beyond `uv`'s own switch; a
child started with an environment of its own making; an interpreter that
already has a `sitecustomize` earlier on its path; and a connection made below
`socket`, by an extension with a resolver of its own.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
from collections.abc import Iterator
from pathlib import Path

LOG = "NETGUARD_LOG"
"""The environment variable naming the file a guarded child records in."""

SITE = Path(__file__).resolve().parent / "_netguard_site"
"""The directory whose `sitecustomize.py` guards a Python child."""

CHILD_VARIABLES = ("PYTHONPATH", LOG, "UV_OFFLINE")
"""What `child_environment` sets."""

# Names that resolve nowhere by definition (RFC 2606 and 6761), and the host
# Starlette's TestClient puts in its URLs. A lookup of one reaches nothing, so
# it is refused and not recorded.
_RESERVED_SUFFIXES = (".test", ".example", ".invalid")
_RESERVED_NAMES = {"testserver", "test", "example", "invalid"}

_attempts: list[str] = []
_installed = False
_permitted = False
_log: str | None = None


def is_loopback(host: object) -> bool:
    """True for a loopback or unspecified address, and for `localhost` names."""
    name = "" if host is None else str(host).rstrip(".").lower()
    if name in ("", "localhost") or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name.split("%")[0])
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def is_reserved(host: object) -> bool:
    """True for a name that resolves nowhere by definition."""
    name = str(host).rstrip(".").lower()
    return name in _RESERVED_NAMES or name.endswith(_RESERVED_SUFFIXES)


def refusal(host: object) -> str | None:
    """Why a lookup or connection to `host` is refused, or None to let it through."""
    if _permitted or is_loopback(host):
        return None
    return f"netguard: nothing past loopback, and {host} is not loopback"


def _record(kind: str, target: object) -> None:
    line = f"{kind} {target}"
    _attempts.append(line)
    if _log:
        with open(_log, "a", encoding="utf-8") as handle:
            handle.write(f"{line} pid={os.getpid()}\n")


def install(log: str | None = None) -> None:
    """Guard this process, recording in memory and, given `log`, in that file.

    A child passes the file its parent reads; the parent reads its own
    attempts from memory. A second call does nothing.
    """
    global _installed, _log
    if _installed:
        return
    _log = log
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        refused = refusal(name)
        if refused:
            if not is_reserved(name):
                _record("lookup", name)
            raise socket.gaierror(socket.EAI_NONAME, refused)
        return real_getaddrinfo(host, *args, **kwargs)

    def _refused(address) -> str | None:
        if isinstance(address, tuple) and address:
            return refusal(address[0])
        return None  # a Unix socket, or another family with no host

    def connect(self, address):
        refused = _refused(address)
        if refused:
            _record("connect", address)
            raise ConnectionRefusedError(refused)
        return real_connect(self, address)

    def connect_ex(self, address):
        if _refused(address):
            _record("connect", address)
            return 111  # ECONNREFUSED
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    _installed = True


def taken() -> list[str]:
    """What this process attempted since the last call, and forget it."""
    found = list(_attempts)
    _attempts.clear()
    return found


@contextlib.contextmanager
def permitted() -> Iterator[None]:
    """Let everything through, and record nothing, inside the block."""
    global _permitted
    saved, _permitted = _permitted, True
    try:
        yield
    finally:
        _permitted = saved


def child_environment(log: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """`environ` (default: this process's) with what guards a Python child."""
    env = dict(os.environ if environ is None else environ)
    path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(SITE), path] if path else [str(SITE)])
    env[LOG] = str(log)
    env["UV_OFFLINE"] = "1"
    return env
