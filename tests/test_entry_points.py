"""qmcp starts however it is started.

`uv run qmcp <command>` is the declared form, and `python -m qmcp` and
`python qmcp` (the directory, from the checkout) start the same CLI. The third
died before printing a line: Python put the package's own directory first on
`sys.path`, and `qmcp/logging.py` replaced the standard library's `logging`
for everything imported after it.

Two guards, because there were two causes. No module here is named after a
standard-library module, and the entry point takes the package directory off
the path before importing anything, so a future collision cannot do it again.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import qmcp

PACKAGE = Path(qmcp.__file__).resolve().parent
CHECKOUT = PACKAGE.parent


def _run(argv, cwd=CHECKOUT, **env):
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", **env}, timeout=120,
    )


def test_no_module_is_named_after_the_standard_library():
    import pkgutil

    names = {m.name for m in pkgutil.iter_modules([str(PACKAGE)])}
    assert not names & set(sys.stdlib_module_names), (
        "a module named after a standard-library module shadows it whenever this "
        "package's directory is on sys.path: " + ", ".join(sorted(names & set(sys.stdlib_module_names))))


@pytest.mark.parametrize("argv", [
    [sys.executable, "-m", "qmcp", "--version"],
    [sys.executable, "qmcp", "--version"],
], ids=["python -m qmcp", "python qmcp"])
def test_every_form_starts_the_cli(argv):
    done = _run(argv)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "qmcp, version" in done.stdout


def test_the_console_script_starts_the_cli():
    script = shutil.which("qmcp", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.skip("no console script beside this interpreter")
    done = _run([script, "--version"])
    assert done.returncode == 0, done.stderr[-2000:]
    assert "qmcp, version" in done.stdout


def test_a_module_named_like_the_standard_library_cannot_shadow_it_when_run_as_a_directory(tmp_path):
    """The collision built on purpose, in a copy: a `json.py` in the package
    that refuses to be imported. Run as `python <copy>/qmcp`, the package's
    directory would put it in place of the standard library's `json`; the entry
    point takes that directory off the path first, so the copy starts."""
    copy = tmp_path / "checkout" / "qmcp"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "json.py").write_text(
        'raise ImportError("the standard library\'s json was shadowed by qmcp/json.py")\n',
        encoding="utf-8")

    done = _run([sys.executable, str(copy), "--version"], cwd=tmp_path)

    assert done.returncode == 0, done.stderr[-2000:]
    assert "qmcp, version" in done.stdout
