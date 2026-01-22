from __future__ import annotations

import importlib
import importlib.util
import os
import runpy
from pathlib import Path

import pytest

FLOW_DIR = Path(__file__).resolve().parents[1] / "examples" / "flows"

FLOW_MODULES = [
    ("simple_plan.py", ["metaflow"]),
    ("approved_deploy.py", ["metaflow"]),
    ("local_agent_chain.py", ["metaflow", "pydantic_ai"]),
    ("local_qc_gauntlet.py", ["metaflow", "pydantic_ai"]),
    ("local_release_notes.py", ["metaflow", "pydantic_ai"]),
]

HELPER_MODULES = [
    ("local_dev_db.py", []),
    ("local_mcp.py", []),
    ("local_llm.py", ["pydantic_ai"]),
]


def _missing_deps(deps: list[str]) -> list[str]:
    return [dep for dep in deps if importlib.util.find_spec(dep) is None]


def _configure_metaflow_env(monkeypatch, tmp_path: Path) -> None:
    metaflow_home = tmp_path / "metaflow"
    metaflow_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("METAFLOW_HOME", str(metaflow_home))
    monkeypatch.setenv("METAFLOW_DEFAULT_DECORATORS", "")


@pytest.mark.parametrize("filename,deps", FLOW_MODULES)
def test_flow_modules_importable(monkeypatch, tmp_path: Path, filename: str, deps: list[str]) -> None:
    missing = _missing_deps(deps)
    if missing:
        pytest.skip(f"Missing optional dependencies: {', '.join(missing)}")

    if "metaflow" in deps and os.name == "nt":
        if importlib.util.find_spec("fcntl") is None:
            pytest.skip("Metaflow plugins require fcntl (POSIX-only). Skipping on Windows.")

    _configure_metaflow_env(monkeypatch, tmp_path)
    monkeypatch.syspath_prepend(str(FLOW_DIR))
    runpy.run_path(str(FLOW_DIR / filename), run_name="__qmcp_test__")


@pytest.mark.parametrize("filename,deps", HELPER_MODULES)
def test_flow_helper_modules_importable(
    monkeypatch,
    tmp_path: Path,
    filename: str,
    deps: list[str],
) -> None:
    missing = _missing_deps(deps)
    if missing:
        pytest.skip(f"Missing optional dependencies: {', '.join(missing)}")

    _configure_metaflow_env(monkeypatch, tmp_path)
    monkeypatch.syspath_prepend(str(FLOW_DIR))
    module_name = Path(filename).stem
    importlib.import_module(module_name)
