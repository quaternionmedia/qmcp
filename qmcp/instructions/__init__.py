"""The instruction inbox: what a person asked for, recorded against a project.

    uv run qmcp instruct "<text>"        record one, typed
    uv run qmcp instruct --voice         speak one
    uv run qmcp instructions list        what has been recorded
    uv run qmcp cookbook instruct        the spoken path, offline

**RECORDING EXECUTES NOTHING.** The voice loop answers questions an agent asks:
a closed choice, spoken and recorded on the human queue. This runs the other
direction -- a person speaks or types an instruction and it is kept, in their
words, with the project it is for -- and it stops there. The row has two
statuses, `recorded` and `unresolved`, and neither says anything is running:
acting on an instruction is a later change, behind consent on the human queue,
with statuses and a migration of its own. Keeping the two apart is what lets a
person speak freely into the inbox: nothing said here spends, writes or runs.

**THE PROJECT IS READ, AND THE READING IS KEPT BESIDE THE CLAIM.** Which project
an instruction is for is decided by `resolve`, by the same whole-word match
`qmcp.threads.consolidate` uses on the thread archive, against the same roster
(`governance/qm`'s own workspace). `rad` is three letters inside `gradient`, so a
substring is not a match; casing is ignored because a transcript capitalises as
it pleases; a name followed by punctuation is still the name. Exactly one match
resolves. None, or several, leaves the instruction `unresolved` with the
candidates in `detail`, where the rule that read them is written beside them --
a project assigned without its evidence would be a guess that looks like a
finding. A caller may state the project outright, which skips the matching and
is recorded as `stated`.

**AN AMBIGUOUS NAME IS ASKED BACK, NEVER GUESSED.** Spoken, an unresolved
instruction becomes a closed choice over the candidates by name, through the
helpers the approval dialog uses and with the same grammar. Whether a project
name transcribes reliably is unmeasured, so the dialog confirms rather than
trusts: the transcript is read back and recorded only on a yes.

**ONE ROSTER, FROM THE CHECKOUT.** The names come from the governance submodule
beside this package rather than from the working directory, because the server
and the subprocess it starts for a spoken instruction must read the same list
whatever directory either was started in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qmcp.db.models import InstructionStatus
from qmcp.threads.consolidate import _pattern, roster

# The governance submodule this checkout carries. Resolved from this file so
# the server and `qmcp instruct --voice`, which the server starts as a process
# of its own, read one roster regardless of either one's working directory.
CORPUS = Path(__file__).resolve().parents[2] / "governance" / "qm"

# The rules `resolve` can report, as data so a reader of `detail` can match
# them rather than parse a sentence.
RULE_STATED = "stated"
RULE_ONE = "the one project named, as a whole word"
RULE_NONE = "no project named"
RULE_SEVERAL = "several projects named"
RULE_NO_ROSTER = "no roster to match against"


@dataclass(frozen=True)
class Resolution:
    """Which project an instruction is for, the names that matched, and the rule.

    Carried together, as `consolidate.Reading` carries its evidence: a project
    without the rule that chose it is a verdict nobody can argue with.
    """

    project: str | None
    candidates: tuple[str, ...]
    rule: str

    @property
    def status(self) -> InstructionStatus:
        return (InstructionStatus.RECORDED if self.project is not None
                else InstructionStatus.UNRESOLVED)

    def detail(self) -> dict:
        """The evidence, as the row keeps it."""
        return {"candidates": list(self.candidates), "rule": self.rule}


def resolve(text: str, names: Iterable[str], project: str | None = None) -> Resolution:
    """Which project `text` names, by whole word, or the one stated.

    `project` given skips the matching entirely: the caller knows, and the
    record says the caller said so. Otherwise every roster name is tried as a
    whole word, case ignored, and exactly one hit resolves.
    """
    if project is not None:
        return Resolution(project=project, candidates=(), rule=RULE_STATED)
    known = tuple(names)
    if not known:
        return Resolution(project=None, candidates=(), rule=RULE_NO_ROSTER)
    named = tuple(name for name in known if _pattern(name).search(text))
    if len(named) == 1:
        return Resolution(project=named[0], candidates=named, rule=RULE_ONE)
    if not named:
        return Resolution(project=None, candidates=(), rule=RULE_NONE)
    return Resolution(project=None, candidates=named, rule=RULE_SEVERAL)


def roster_names(corpus: Path = CORPUS) -> tuple[str, ...]:
    """The repository names the corpus's workspace declares, in its order.

    Empty when the submodule is not checked out: an instruction is then
    recorded `unresolved` with `RULE_NO_ROSTER`, which is a different answer
    from "no project named" and is kept distinct on purpose.
    """
    if not (corpus / "ci" / "workspace.yaml").is_file():
        return ()
    return tuple(roster(corpus))


__all__ = [
    "CORPUS",
    "RULE_NONE",
    "RULE_NO_ROSTER",
    "RULE_ONE",
    "RULE_SEVERAL",
    "RULE_STATED",
    "Resolution",
    "resolve",
    "roster_names",
]
