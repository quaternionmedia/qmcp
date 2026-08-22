"""What a command is allowed to spend, and what it says when it spent nothing.

**THIS MODULE DOES NOT CALL A PAID SERVICE. It is what a module that does must
pass through**, and it exists so that a reader can find out whether a command
spends without tracing a call graph.
`governance/qm/records/DRAFT-no-unattended-spending.md` is the decision.

THE RULE, IN ONE SENTENCE. No unattended process may call a paid service; every
paid call is a direct, deterministic, human-issued command.

ZERO IS THE DEFAULT AND IT IS A REAL COUNT. A command issued with a budget of
zero does every free thing it can, establishes what the paid work would cost,
and stops. That is how a count gets stated in advance without spending to find
it out -- the first pass is free, and the number it produces is what a second
pass is issued against.

ZERO IS NEVER A SENTINEL FOR UNKNOWN, and the distinction is load-bearing:

    authorised   how many calls the person permitted. Always known.
    made         how many were made. Always known.
    would_need   how many the work requires. May be `{"unknown": reason}`.

Reporting `would_need: 0` when nobody could count would claim the work is free.
That is the substitution `harness-status.json`'s reading block refuses and the
harness payload already avoids, arriving here in a new register.

WHAT THIS CANNOT DO. Stop a module that never asked it. Clause 6 of the record
is a rule about writing rather than a derivation, and a paid surface added
without declaring it is the failure this depends on people not committing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The default, and it is not a placeholder. A command nobody gave a number to
# has been given zero, which is a budget rather than an absence of one.
FREE = 0


def unknown(reason: str) -> dict[str, str]:
    """A requirement nobody could establish, and why. Never zero."""
    return {"unknown": reason}


class Refused(Exception):
    """A call that would exceed what a person authorised.

    Raised rather than returned: a caller that can ignore the answer will, and
    the thing being protected is somebody's money.
    """


@dataclass
class Budget:
    """What one command may spend, and what it has spent.

    A budget is per command. It is not stored, not remembered between runs, and
    has no way to be widened after the fact -- clause 5 of the record, because
    what a person consents to is an amount rather than a category.
    """

    authorised: int = FREE
    made: int = 0
    service: str | None = None

    def __post_init__(self) -> None:
        if self.authorised < 0:
            raise ValueError(
                f"authorised={self.authorised}: a negative budget is not a "
                f"smaller one. Zero is the floor and it is a real count."
            )

    @property
    def free(self) -> bool:
        """True when this command was issued to spend nothing."""
        return self.authorised == FREE

    @property
    def remaining(self) -> int:
        return max(0, self.authorised - self.made)

    def spend(self, calls: int = 1) -> None:
        """Record `calls` about to happen, or refuse.

        Checked before the call rather than after, because a refusal that
        arrives after the money is gone is a report.
        """
        if calls < 1:
            raise ValueError("a call costs at least one call")
        if self.made + calls > self.authorised:
            raise Refused(
                f"{self.made + calls} call(s) would exceed the {self.authorised} "
                f"authorised for this command"
                + (f" against {self.service}" if self.service else "")
                + ". Re-issue it with the count you mean to permit; nothing "
                  "here widens a budget it was given."
            )
        self.made += calls


def declare(budget: Budget, would_need: int | dict[str, str] | None) -> dict[str, Any]:
    """What this run spent, and what the work would cost, for a consumer.

    CARRIED WITH THE PAYLOAD, NOT INFERRED FROM IT. A consumer must be able to
    tell a free-path result from a complete one without guessing at an empty
    field: they are different claims, and a partial answer presented as whole
    is the shape of finding this organisation keeps recording.
    """
    if would_need is None:
        would_need = unknown("this run did not establish what the work costs")
    if isinstance(would_need, int) and would_need < 0:
        raise ValueError("a requirement of fewer than zero calls is not a count")

    return {
        "authorised": budget.authorised,
        "made": budget.made,
        "would_need": would_need,
        "service": budget.service,
        # Said outright rather than left to be derived from `made == 0`, which
        # is also true of a run that was authorised and found nothing to do.
        "free_pass": budget.free,
    }


def render(declared: dict[str, Any]) -> str:
    """The spend, for a person deciding whether to issue the paid run."""
    need = declared.get("would_need")
    service = declared.get("service") or "a paid service"

    if declared.get("free_pass"):
        lines = [f"  Nothing was spent. This pass was issued against {FREE} calls."]
        if isinstance(need, dict) and "unknown" in need:
            lines += [
                f"  What the paid work would cost is unknown: {need['unknown']}",
                "  A count nobody took is not a count of zero, so this does not",
                "  say the work is free.",
            ]
        elif need:
            lines += [
                f"  The paid work would need {need} call(s) against {service}.",
                f"  Re-issue with that number to permit them. Consent does not",
                f"  carry forward, so this is asked every time.",
            ]
        else:
            lines.append("  There is no paid work to do.")
        return "\n".join(lines)

    return "\n".join([
        f"  {declared['made']} of {declared['authorised']} authorised call(s) "
        f"made against {service}.",
        "  This budget was for this command and is not remembered.",
    ])
