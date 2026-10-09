"""A consent-gated, read-only runtime for the runnable cross-check topology."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from qmcp import orchestration as plane
from qmcp.agentframework.models.enums import TopologyType
from qmcp.integrations.agents import AgentOutcome, AgentRuntime, Brief, OnEvent
from qmcp.orchestration import cross_check

NAME = "topology-crosscheck"
CHECKER_STEPS = 3
FINAL_RESPONSE_REQUESTS = 1
MAX_REQUESTS_PER_CHECKER = CHECKER_STEPS + FINAL_RESPONSE_REQUESTS
MAX_CHAT_REQUESTS = 3 * MAX_REQUESTS_PER_CHECKER
MAX_TOKENS_PER_REQUEST = 512
REQUEST_TIMEOUT_SECONDS = 90

PERSPECTIVES = (
    "Look for direct evidence in project files that supports the claim.",
    "Actively look for project evidence or edge cases that contradict the claim.",
    "Check the claim's assumptions against the implementation and its tests.",
    "Check whether the claim is precise, scoped correctly, and supported by evidence.",
    "Look for failure paths and boundary conditions that could falsify the claim.",
    "Check whether tests provide evidence or merely repeat the claim's assumptions.",
    "Check whether configuration, defaults, or environment change the behavior.",
    "Look for compatibility and integration evidence that contradicts the claim.",
    "Check for a mismatch between the implementation and its documentation.",
    "Independently identify evidence missing to support the claim.",
)


def checker_prompt(perspective: str, claim: str) -> str:
    """The complete task instruction given to one isolated checker."""
    return "\n".join((
        perspective,
        "Assess whether the claim is supported by files in this project.",
        "Read project files only. Do not run commands, change files, or take action.",
        "The first line must be exactly VERDICT: YES or VERDICT: NO.",
        "The second line must be exactly EVIDENCE: <project-relative file path> | "
        "<one concise reason>.",
        "Read that cited file with read_file before returning the verdict.",
        f"Claim: {claim}",
    ))


def _parse_verdict(text: str) -> tuple[bool | None, str, str | None]:
    """Accept only the declared verdict and file-citation result shape."""
    lines = (text or "").strip().splitlines()
    if len(lines) < 2:
        return None, "The checker did not return a verdict and file evidence.", None
    verdict = re.fullmatch(r"VERDICT:\s*(YES|NO)", lines[0].strip(), flags=re.IGNORECASE)
    evidence = re.fullmatch(
        r"EVIDENCE:\s*(.*?)\s*\|\s*(.+)", lines[1].strip(), flags=re.IGNORECASE
    )
    if verdict is None or evidence is None:
        return None, "The checker response did not match the required verdict format.", None
    path = evidence.group(1).strip().strip("`<>").replace("\\", "/").removeprefix("./")
    reason = evidence.group(2).strip()
    if not path or not reason:
        return None, "The checker response did not name a file and a reason.", None
    return verdict.group(1).upper() == "YES", f"{path}: {reason}", path


def _read_file_paths(reads: Any) -> set[str]:
    """Extract project-relative paths actually passed to the read-file tool."""
    if not isinstance(reads, list):
        return set()
    paths = set()
    for invocation in reads:
        if not isinstance(invocation, str):
            continue
        match = re.fullmatch(r"read_file\((.+)\)", invocation)
        if match:
            paths.add(match.group(1).replace("\\", "/").removeprefix("./"))
    return paths


def _local_runtime() -> AgentRuntime:
    """A local reader with retries disabled so each request consumes the cap."""
    from qmcp.integrations.agents.adapters.ollama import Runtime

    return Runtime(
        max_steps=CHECKER_STEPS,
        recover_timeouts=False,
        max_tokens=MAX_TOKENS_PER_REQUEST,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class CrossCheckRuntime:
    """Run three independent local checkers and report their votes, not a decision."""

    name = NAME
    would_spend = 0
    uses_history = False

    def __init__(
        self,
        prompt: str,
        runtime_factory: Callable[[], AgentRuntime] = _local_runtime,
        *,
        perspectives: tuple[str, ...] | None = None,
        checker_count: int | None = None,
    ) -> None:
        self.prompt = prompt
        self.runtime_factory = runtime_factory
        count = checker_count or (
            len(perspectives) if perspectives else 3
        )
        if not 2 <= count <= 10:
            raise ValueError("crosscheck needs between 2 and 10 checkers")
        self.perspectives = perspectives or PERSPECTIVES[:count]
        if len(self.perspectives) != count:
            raise ValueError("crosscheck perspective count does not match checker_count")
        self.max_chat_requests = count * MAX_REQUESTS_PER_CHECKER
        self.command = self._consent_description()

    def _consent_description(self) -> str:
        """The topology, exact checker prompts and enforced inference ceiling."""
        prompts = [
            f"Checker {number}: {checker_prompt(perspective, self.prompt)}"
            for number, perspective in enumerate(self.perspectives, start=1)
        ]
        prompts.extend((
            f"Budget: one topology run; at most {self.max_chat_requests} "
            "local-model chat requests total, "
            "four per checker including one final-answer request; each request "
            "is limited to 512 generated tokens and 90 seconds.",
            "Each checker starts independently with no earlier instruction history. "
            "The local runtime has read-only project-file tools and no shell or write tool.",
        ))
        return "Topology: crosscheck. " + " ".join(prompts)

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        """Run only while the capability plane still declares this shape safe."""
        capability = plane.by_type().get(TopologyType.CROSS_CHECK)
        refusal = plane.refuses(TopologyType.CROSS_CHECK, "cross-check a claim")
        if (
            capability is None
            or not capability.can_run
            or capability.spends
            or capability.writes
            or capability.decides
            or refusal
            or TopologyType.CROSS_CHECK.value in plane.stubs()
        ):
            reason = refusal or "the capability plane no longer declares crosscheck safe to run"
            return AgentOutcome(
                text=f"Cross-check refused: {reason}.",
                exit_code=1,
                spent=0,
                detail={"topology": TopologyType.CROSS_CHECK.value, "model_requests": 0},
            )

        began = time.monotonic()
        results_by_checker: dict[int, dict[str, Any]] = {}

        def run_checker(index: int, perspective: str) -> Callable[[str], tuple[bool, str]]:
            def check(claim: str) -> tuple[bool, str]:
                try:
                    runtime = self.runtime_factory()
                    instruction = checker_prompt(perspective, claim)
                    checker_brief = Brief(
                        instruction=instruction,
                        cwd=brief.cwd,
                        project=brief.project,
                    )

                    def progress(state: str, message: str) -> None:
                        if on_event:
                            on_event(state, f"cross-checker: {message}")

                    outcome = runtime.run(checker_brief, on_event=progress)
                except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    results_by_checker[index] = {
                        "perspective": perspective,
                        "verdict": None,
                        "reason": reason,
                        "text": "",
                        "model_calls": None,
                    }
                    return False, f"checker failed: {reason}"

                reported_calls = outcome.detail.get("model_calls")
                known_call_count = (
                    isinstance(reported_calls, int)
                    and not isinstance(reported_calls, bool)
                    and reported_calls >= 0
                )
                within_call_budget = (
                    known_call_count and reported_calls <= MAX_REQUESTS_PER_CHECKER
                )
                verdict, evidence, cited_path = (
                    _parse_verdict(outcome.text)
                    if outcome.succeeded
                    else (
                        None,
                        outcome.text or "The local model run failed without a result.",
                        None,
                    )
                )
                reported_reads = outcome.detail.get("read")
                read_files = _read_file_paths(reported_reads)
                if not outcome.succeeded:
                    verdict = None
                elif not within_call_budget:
                    verdict, evidence = (
                        None,
                        "The runtime did not report a model-call count within this checker's "
                        "budget of four requests.",
                    )
                elif outcome.spent != 0:
                    verdict, evidence = (
                        None,
                        "The local checker reported provider spending; its response is not "
                        "counted.",
                    )
                elif not read_files:
                    verdict, evidence = (
                        None,
                        "No project file was read; this response is not counted.",
                    )
                elif cited_path not in read_files:
                    verdict, evidence = (
                        None,
                        f"The cited file {cited_path!r} was not read; this response is not "
                        "counted.",
                    )
                results_by_checker[index] = {
                    "perspective": perspective,
                    "verdict": verdict,
                    "reason": evidence,
                    "text": outcome.text,
                    "model_calls": reported_calls if known_call_count else None,
                    "reads": reported_reads if isinstance(reported_reads, list) else None,
                }
                if verdict is None:
                    return False, evidence
                return verdict, evidence

            return check

        found = cross_check(
            self.prompt,
            [run_checker(index, perspective)
             for index, perspective in enumerate(self.perspectives)],
        )
        results = []
        for index, perspective in enumerate(self.perspectives):
            result = results_by_checker.get(index)
            if result is None:
                result = {
                    "perspective": perspective,
                    "verdict": None,
                    "reason": found.reasons[index],
                    "text": "",
                    "model_calls": None,
                }
            results.append(result)
        completed = sum(result["verdict"] is not None for result in results)
        model_calls = sum(
            result["model_calls"] or 0 for result in results
            if isinstance(result["model_calls"], int)
        )
        unknown_call_counts = sum(result["model_calls"] is None for result in results)
        lines = [
                f"Cross-check reports {found.agreed} of {len(self.perspectives)} checkers "
                "support the claim.",
            "This is a report, not a decision; no project files were changed.",
        ]
        if unknown_call_counts:
            lines.append(
                f"{model_calls} model request(s) were reported; the exact total is unknown "
                f"for {unknown_call_counts} checker(s)."
            )
        else:
            lines.append(
                f"Model requests: {model_calls} of at most {self.max_chat_requests}."
            )
        for number, result in enumerate(results, start=1):
            verdict = result["verdict"]
            answer = "YES" if verdict is True else "NO" if verdict is False else "NO VALID VERDICT"
            lines.append(f"{number}. {answer}: {result['reason']}")
        if completed != len(self.perspectives):
            lines.append(
                f"Incomplete: {len(self.perspectives) - completed} checker(s) failed or returned "
                "an invalid response; they are not counted as NO votes."
            )
        return AgentOutcome(
            text="\n".join(lines),
            exit_code=0 if completed == len(self.perspectives) else 1,
            elapsed_seconds=round(time.monotonic() - began, 2),
            spent=0,
            detail={
                "topology": TopologyType.CROSS_CHECK.value,
                "checker_count": len(self.perspectives),
                "completed_checkers": completed,
                "model_calls": model_calls,
                "unknown_model_call_counts": unknown_call_counts,
                "max_model_calls": self.max_chat_requests,
                "checkers": results,
            },
        )


__all__ = [
    "CHECKER_STEPS",
    "CrossCheckRuntime",
    "MAX_CHAT_REQUESTS",
    "MAX_REQUESTS_PER_CHECKER",
    "MAX_TOKENS_PER_REQUEST",
    "PERSPECTIVES",
    "REQUEST_TIMEOUT_SECONDS",
    "checker_prompt",
]
