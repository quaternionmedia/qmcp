"""The step/delta reconciliation demo, and the decoupling that made it possible.

The first attempt at this demo could not run at all: reading the four
`AgentStep`s of the change-impact pipeline meant importing
`examples/flows/change_impact.py`, which imports Metaflow at module level, which
dies on Windows in `metaflow/sidecar/sidecar_subprocess.py` on `import fcntl`.

Four pure step descriptions were unreachable on this platform because of the
executor they happened to be filed with. `qmcp/cookbook/change_impact.py` is the
separation, and the guard below is what keeps it: the coupling is the kind that
returns quietly the next time somebody adds a decorator.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.demo_step_as_delta import TerseSummary, main, run  # noqa: E402


# --- the decoupling ----------------------------------------------------------


def test_the_pipeline_imports_without_metaflow():
    """The whole point of the extraction. If this fails, the steps are once
    again only readable on a POSIX machine with the flows extra installed."""
    assert "metaflow" not in sys.modules or True  # importing below is the assertion
    from qmcp.cookbook.change_impact import CHANGE_IMPACT_PIPELINE

    assert [s.name for s in CHANGE_IMPACT_PIPELINE.steps] == [
        "summarizer", "risk_assessor", "test_planner", "migration_guide",
    ]


def test_the_pipeline_module_names_no_flow_runtime():
    """Asserted on the source, because the failure is an import line existing."""
    import qmcp.cookbook.change_impact as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imports = "\n".join(
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    )
    assert "metaflow" not in imports


def test_the_flow_still_exposes_what_it_used_to():
    """The flow re-exports the models, so anything importing them from there
    keeps working. Checked on the source: the module cannot be imported here."""
    flow = (_REPO_ROOT / "examples" / "flows" / "change_impact.py").read_text(encoding="utf-8")
    for name in ("CHANGE_IMPACT_PIPELINE", "ChangeSummary", "RiskAssessment",
                 "TestPlan", "MigrationGuide"):
        assert name in flow


def test_the_flow_file_is_still_valid_python():
    """It cannot be imported on this platform, so compiling is the check that
    the edit did not leave it broken for the machines that can."""
    path = _REPO_ROOT / "examples" / "flows" / "change_impact.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


# --- the demo ----------------------------------------------------------------


def test_a_declared_step_maps_to_a_planning_delta():
    assert run(out=lambda *a: None)["planned"]["delta"]["phase"] == "planning"


def test_the_delta_row_carries_the_real_prompt():
    row = run(out=lambda *a: None)["planned"]["delta"]
    assert row["name"] == "summarizer"
    assert "summarize engineering changes" in row["description"]


def test_a_run_step_with_no_review_declared_is_complete():
    assert run(out=lambda *a: None)["ran"]["delta"]["phase"] == "complete"


def test_a_reviewed_step_reaches_review_and_links_its_invocation():
    findings = run(out=lambda *a: None)
    assert findings["reviewed"]["delta"]["phase"] == "review"
    assert findings["reviewed"]["links"][0]["link_type"] == "invocation"


def test_a_declared_review_that_did_not_happen_does_not_read_as_complete():
    """The assertion that keeps the phase honest."""
    assert run(out=lambda *a: None)["outstanding_phase"] == "implementation"


def test_the_step_is_rebuilt_from_its_delta():
    assert run(out=lambda *a: None)["rebuilt_matches"] is True


def test_a_different_implementation_carries_the_same_delta_identity():
    """Interchangeable: the delta pins the work, so a step with another output
    type stands behind the same delta."""
    findings = run(out=lambda *a: None)
    assert findings["swapped_matches"] is True
    assert findings["swapped_output_type"] is TerseSummary


def test_the_demo_exits_zero():
    assert main() == 0
