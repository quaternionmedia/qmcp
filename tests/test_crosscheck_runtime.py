"""The voice CrossCheck topology runs only the local, bounded, read-only path."""

from __future__ import annotations

from qmcp.integrations.agents import AgentOutcome, Brief, crosscheck
from qmcp.integrations.agents.adapters import ollama


def _outcome(text: str, *, calls: int = 1, code: int = 0,
             reads: list[str] | None = None) -> AgentOutcome:
    return AgentOutcome(
        text=text,
        exit_code=code,
        spent=0,
        detail={
            "model_calls": calls,
            "read": reads if reads is not None else ["read_file(README.md)"],
        },
    )


class _FakeRuntime:
    def __init__(self, outcome: AgentOutcome, briefs: list[Brief]) -> None:
        self.outcome = outcome
        self.briefs = briefs

    def run(self, brief: Brief, on_event=None) -> AgentOutcome:
        self.briefs.append(brief)
        return self.outcome


def _runner(outcomes: list[AgentOutcome], briefs: list[Brief]):
    made = 0

    def factory():
        nonlocal made
        outcome = outcomes[made]
        made += 1
        return _FakeRuntime(outcome, briefs)

    return factory


def test_three_independent_checkers_report_votes_and_the_actual_request_total(tmp_path):
    briefs: list[Brief] = []
    runtime = crosscheck.CrossCheckRuntime(
        "the tools cannot modify files",
        runtime_factory=_runner(
            [
                _outcome(
                    "VERDICT: YES\nEVIDENCE: README.md | tools are read-only",
                    calls=4,
                ),
                _outcome(
                    "VERDICT: NO\nEVIDENCE: qmcp/server.py | a write route exists",
                    calls=4,
                    reads=["read_file(qmcp/server.py)"],
                ),
                _outcome(
                    "VERDICT: YES\nEVIDENCE: tests/test_readonly.py | tests reject writes",
                    calls=4,
                    reads=["read_file(tests/test_readonly.py)"],
                ),
            ],
            briefs,
        ),
    )

    outcome = runtime.run(Brief(
        instruction="outer instruction",
        cwd=tmp_path,
        project="qmcp",
    ))

    assert outcome.succeeded
    assert "2 of 3 checkers support the claim" in outcome.text
    assert "Model requests: 12 of at most 12." in outcome.text
    assert outcome.detail["model_calls"] == 12
    assert outcome.detail["max_model_calls"] == 12
    assert outcome.detail["unknown_model_call_counts"] == 0
    assert len(briefs) == 3
    for brief, perspective in zip(briefs, crosscheck.PERSPECTIVES[:3]):
        assert brief.project == "qmcp" and brief.cwd == tmp_path
        assert brief.history == ()
        assert brief.instruction == crosscheck.checker_prompt(
            perspective, "the tools cannot modify files"
        )


def test_saved_design_crosscheck_uses_its_reusable_checker_perspectives(tmp_path):
    briefs: list[Brief] = []
    runtime = crosscheck.CrossCheckRuntime(
        "the claim",
        runtime_factory=_runner(
            [
                _outcome("VERDICT: YES\nEVIDENCE: README.md | supports it"),
                _outcome(
                    "VERDICT: NO\nEVIDENCE: qmcp/server.py | contradicts it",
                    reads=["read_file(qmcp/server.py)"],
                ),
            ],
            briefs,
        ),
        perspectives=("Support-checker instruction.", "Critic instruction."),
        checker_count=2,
    )
    outcome = runtime.run(Brief("outer", tmp_path, "qmcp"))
    assert outcome.succeeded
    assert outcome.detail["checker_count"] == 2
    assert outcome.detail["max_model_calls"] == 8
    assert "Cross-check reports 1 of 2 checkers" in outcome.text
    assert "Critic instruction." in briefs[1].instruction
    assert len(briefs) == 2


def test_malformed_or_failed_checkers_are_not_counted_as_no_votes(tmp_path):
    briefs: list[Brief] = []
    runtime = crosscheck.CrossCheckRuntime(
        "the tools cannot modify files",
        runtime_factory=_runner(
            [
                _outcome("VERDICT: YES\nEVIDENCE: README.md | supported"),
                _outcome("I think yes, probably"),
                _outcome("model stopped", code=1),
            ],
            briefs,
        ),
    )

    outcome = runtime.run(Brief("claim", tmp_path, "qmcp"))

    assert outcome.exit_code == 1
    assert outcome.detail["completed_checkers"] == 1
    assert "Incomplete: 2 checker(s)" in outcome.text
    assert "NO VALID VERDICT" in outcome.text
    assert "2 of 3" not in outcome.text


def test_an_over_budget_checker_is_reported_as_a_failure_with_its_real_count(tmp_path):
    briefs: list[Brief] = []
    runtime = crosscheck.CrossCheckRuntime(
        "the tools cannot modify files",
        runtime_factory=_runner(
            [
                _outcome(
                    "VERDICT: YES\nEVIDENCE: README.md | supported",
                    calls=5,
                ),
                _outcome(
                    "VERDICT: NO\nEVIDENCE: qmcp/server.py | unsupported",
                    reads=["read_file(qmcp/server.py)"],
                ),
                _outcome(
                    "VERDICT: YES\nEVIDENCE: tests/test_readonly.py | supported",
                    reads=["read_file(tests/test_readonly.py)"],
                ),
            ],
            briefs,
        ),
    )

    outcome = runtime.run(Brief("claim", tmp_path, "qmcp"))

    assert outcome.exit_code == 1
    assert outcome.detail["model_calls"] == 7
    assert "four requests" in outcome.text
    assert outcome.detail["completed_checkers"] == 2


def test_a_checker_must_read_the_file_it_cites(tmp_path):
    briefs: list[Brief] = []
    runtime = crosscheck.CrossCheckRuntime(
        "claim",
        runtime_factory=_runner(
            [
                _outcome(
                    "VERDICT: YES\nEVIDENCE: README.md | supported",
                    reads=[],
                ),
                _outcome(
                    "VERDICT: NO\nEVIDENCE: server.py | unsupported",
                    reads=["read_file(server.py)"],
                ),
                _outcome(
                    "VERDICT: YES\nEVIDENCE: README.md | supported",
                    reads=["read_file(server.py)"],
                ),
            ],
            briefs,
        ),
    )

    outcome = runtime.run(Brief("claim", tmp_path, "qmcp"))

    assert outcome.exit_code == 1
    assert outcome.detail["completed_checkers"] == 1
    assert "No project file was read" in outcome.text
    assert "The cited file 'README.md' was not read" in outcome.text
    assert outcome.detail["checkers"][1]["reads"] == ["read_file(server.py)"]


def test_runtime_factory_enforces_the_consented_request_limits(monkeypatch):
    received: dict[str, object] = {}

    class LocalRuntime:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(ollama, "Runtime", LocalRuntime)
    crosscheck.CrossCheckRuntime("claim").runtime_factory()

    assert received == {
        "max_steps": crosscheck.CHECKER_STEPS,
        "recover_timeouts": False,
        "max_tokens": crosscheck.MAX_TOKENS_PER_REQUEST,
        "timeout": crosscheck.REQUEST_TIMEOUT_SECONDS,
    }


def test_a_topology_refusal_makes_no_local_model_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(crosscheck.plane, "refuses", lambda *_: "refused for this test")

    def unexpected_model():
        raise AssertionError("refused topology must not construct a local model")

    outcome = crosscheck.CrossCheckRuntime(
        "claim", runtime_factory=unexpected_model
    ).run(Brief("claim", tmp_path, "qmcp"))

    assert outcome.exit_code == 1
    assert "Cross-check refused:" in outcome.text
    assert outcome.detail["model_requests"] == 0


def test_an_unexpected_checker_exception_is_visible_and_not_a_vote(tmp_path):
    def broken_runtime():
        raise TypeError("invalid test runtime")

    outcome = crosscheck.CrossCheckRuntime(
        "claim", runtime_factory=broken_runtime
    ).run(Brief("claim", tmp_path, "qmcp"))

    assert outcome.exit_code == 1
    assert outcome.detail["completed_checkers"] == 0
    assert outcome.detail["unknown_model_call_counts"] == 3
    assert "TypeError: invalid test runtime" in outcome.text
    assert "exact total is unknown for 3 checker(s)" in outcome.text
