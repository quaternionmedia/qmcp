"""Nothing the suite runs reaches past loopback, and an attempt fails the test.

`qmcp.testing.plugin` installs `qmcp.testing.netguard` before collection and
hands every Python child an environment that installs it too. These tests make
the attempts a test must never make and show each is refused and recorded.
Each takes back what it recorded with `netguard.taken()`, so the plugin's own
check does not fail it; the last two run a pytest of their own to show that
check failing a test.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import httpx
import pytest

from qmcp.testing import netguard


def test_a_lookup_past_loopback_is_refused_and_recorded() -> None:
    # Mutation seen red: `socket.getaddrinfo = getaddrinfo` removed from
    # `install` -- the lookup went to the machine's resolver, and nothing
    # was recorded.
    with pytest.raises(socket.gaierror, match="netguard"):
        socket.getaddrinfo("example.com", 443)
    assert netguard.taken() == ["lookup example.com"]


def test_a_refusal_the_code_catches_is_still_recorded() -> None:
    try:
        httpx.get("https://example.com", timeout=2)
    except httpx.HTTPError:
        pass
    assert netguard.taken() == ["lookup example.com"]


def test_a_connection_to_an_address_past_loopback_is_refused() -> None:
    # Mutation seen red: `socket.socket.connect = connect` removed from
    # `install` -- the connection went out and timed out instead.
    # 192.0.2.1 is TEST-NET-1 (RFC 5737), routed nowhere.
    with socket.socket() as sock, pytest.raises(ConnectionRefusedError, match="netguard"):
        sock.settimeout(1)
        sock.connect(("192.0.2.1", 80))
    assert netguard.taken() == ["connect ('192.0.2.1', 80)"]


def test_loopback_is_let_through() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
    assert netguard.taken() == []


def test_the_machines_own_service_ports_are_refused_on_loopback_too() -> None:
    # Mutation seen red: `install` called by the plugin with no
    # `machine_ports` -- the connection to qmcp's default port went out to
    # whatever listens there; here nothing did, and it timed out unrecorded.
    from qmcp.localmodel import ENDPOINT
    from qmcp.testing.plugin import machine_ports

    ports = machine_ports()
    assert set(ports) == {3141, 8000}
    assert netguard.refusal("127.0.0.1", int(ENDPOINT.rsplit(":", 1)[1])) is None
    for port in ports:
        with socket.socket() as sock, pytest.raises(ConnectionRefusedError, match="netguard"):
            sock.settimeout(1)
            sock.connect(("127.0.0.1", port))
    assert netguard.taken() == [f"connect ('127.0.0.1', {port})" for port in ports]


def test_a_name_reserved_to_resolve_nowhere_is_refused_and_not_recorded() -> None:
    with pytest.raises(socket.gaierror, match="netguard"):
        socket.getaddrinfo("alfred.test", 80)
    assert netguard.taken() == []


def test_permitted_lets_everything_through_and_restores_the_guard() -> None:
    with netguard.permitted():
        assert netguard.refusal("example.com") is None
    assert netguard.refusal("example.com")


def test_the_session_hands_its_children_the_guard() -> None:
    # Mutation seen red: the plugin's `os.environ.update(...)` removed -- the
    # inherited environment carried no guard.
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(netguard.SITE)
    assert os.path.isfile(os.environ[netguard.LOG])
    assert "3141" in os.environ[netguard.PORTS].split(",")
    assert os.environ["UV_OFFLINE"] == "1"


def test_a_python_child_is_guarded_and_records_in_its_log(tmp_path) -> None:
    # Mutation seen red: `_netguard.install()` removed from the child's
    # `sitecustomize.py` -- the child's lookup went to the resolver, and its
    # log stayed empty.
    log = tmp_path / "child.log"
    child = subprocess.run(
        [sys.executable, "-c", "import socket; socket.getaddrinfo('example.com', 443)"],
        env=netguard.child_environment(log), capture_output=True, text=True, timeout=60)

    assert child.returncode != 0
    assert "netguard" in child.stderr
    assert log.read_text(encoding="utf-8").startswith("lookup example.com pid=")


def test_a_python_child_is_refused_the_machines_service_ports(tmp_path) -> None:
    # Mutation seen red: the child's `sitecustomize.py` installing with no
    # ports -- the child dialed 3141 and was answered by the operating system,
    # not refused by the guard.
    log = tmp_path / "child.log"
    child = subprocess.run(
        [sys.executable, "-c",
         "import socket; s = socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 3141))"],
        env=netguard.child_environment(log), capture_output=True, text=True, timeout=60)

    assert "netguard: port 3141" in child.stderr
    assert log.read_text(encoding="utf-8").startswith("connect ('127.0.0.1', 3141) pid=")


INNER = '''
import httpx
import pytest

from qmcp.testing import netguard


def test_swallows_a_refusal():
    try:
        httpx.get("https://example.com", timeout=2)
    except httpx.HTTPError:
        pass


@pytest.mark.live_anthropic
def test_marked_live():
    assert netguard.refusal("example.com") is None


def test_unmarked():
    assert netguard.refusal("example.com")
'''


def _inner_pytest(tmp_path) -> subprocess.CompletedProcess:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(INNER, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "qmcp.testing.plugin", "-p", "no:cacheprovider",
         "-q", "-rA", "test_inner.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)


def test_the_plugin_fails_a_test_whose_code_swallowed_a_refusal(tmp_path) -> None:
    # Mutation seen red: the plugin's `pytest.fail(...)` removed -- the inner
    # test passed with its attempt unreported, and the run exited 0.
    run = _inner_pytest(tmp_path)

    assert run.returncode == 1, run.stdout
    assert "ERROR test_inner.py::test_swallows_a_refusal" in run.stdout
    assert "reached past loopback" in run.stdout
    assert "lookup example.com" in run.stdout


def test_the_plugin_lets_a_live_test_through_and_guards_the_rest(tmp_path) -> None:
    # Mutation seen red: `ONLINE_MARKERS` emptied -- the marked test found the
    # guard in place and failed.
    run = _inner_pytest(tmp_path)

    assert "PASSED test_inner.py::test_marked_live" in run.stdout
    assert "PASSED test_inner.py::test_unmarked" in run.stdout
