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
    ("council_deliberation.py", ["metaflow", "pydantic_ai"]),
    ("qc_release.py", ["metaflow", "pydantic_ai"]),
    ("plan_council.py", ["metaflow", "pydantic_ai"]),
    ("change_impact.py", ["metaflow", "pydantic_ai"]),
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


def test_cookbook_package_importable() -> None:
    """Verify qmcp.cookbook package imports successfully."""
    from qmcp.cookbook import (
        LocalLLMConfig,
        build_local_agent,
        FlowPersistence,
        MCPToolInvoker,
    )
    assert callable(build_local_agent)
