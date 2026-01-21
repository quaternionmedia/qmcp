from __future__ import annotations

import importlib.util
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


@pytest.mark.parametrize("filename,deps", FLOW_MODULES)
def test_flow_modules_importable(monkeypatch, filename: str, deps: list[str]) -> None:
    missing = _missing_deps(deps)
    if missing:
        pytest.skip(f"Missing optional dependencies: {', '.join(missing)}")

    monkeypatch.syspath_prepend(str(FLOW_DIR))
    runpy.run_path(str(FLOW_DIR / filename), run_name="__qmcp_test__")


@pytest.mark.parametrize("filename,deps", HELPER_MODULES)
def test_flow_helper_modules_importable(
    monkeypatch,
    filename: str,
    deps: list[str],
) -> None:
    missing = _missing_deps(deps)
    if missing:
        pytest.skip(f"Missing optional dependencies: {', '.join(missing)}")

    monkeypatch.syspath_prepend(str(FLOW_DIR))
    runpy.run_path(str(FLOW_DIR / filename), run_name="__qmcp_test__")
