"""The instruction inbox: what a person asked for, recorded against a project.

    uv run qmcp instruct "<text>"        record one, typed
    uv run qmcp instruct --voice         speak one
    uv run qmcp instructions list        what has been recorded
    uv run qmcp cookbook instruct        the spoken path, offline

Recording executes nothing. An instruction is kept in the person's words with
the project it is for, at status `recorded` or `unresolved`; neither says
anything is running. Acting on one is a separate command
(`qmcp.instructions.act`), behind consent on the human queue.

`resolve` reads which project an instruction is for, by the whole-word match
`qmcp.threads.consolidate` uses, against the roster in `governance/qm`: case
is ignored, a substring inside another word is not a match, and a hyphenated
name also matches with its hyphens as whitespace, as a transcript carries it.
Exactly one match resolves. None or several leaves the instruction
`unresolved`, with the candidates and the rule in `detail`. A caller may state
the project, recorded as `stated`; a blank states nothing.

The roster is read from the governance submodule beside this package, so the
server and the process it starts for a spoken instruction read one list
whatever their working directories.

`docs/integrations/voice.md` ("An instruction is recorded, not run") describes
the commands, the routes and the spoken dialog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qmcp.db.models import InstructionStatus
from qmcp.threads.consolidate import roster

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

    Carried together, as `consolidate.Reading` carries its evidence, so a
    project always comes with the rule that chose it.
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


def _mention(name: str) -> re.Pattern[str]:
    """A repository name as a whole word, written or as a transcript says it.

    The boundaries are `qmcp.threads.consolidate._pattern`'s, so `rad` is not
    found inside `gradient` or `rad-godot`. Each hyphen in the name also
    matches whitespace, so `rad godot` is `rad-godot`: a transcript carries
    no hyphens, and the spoken form is the only one a dialog ever hears.
    """
    body = r"[\s-]+".join(re.escape(part) for part in name.split("-"))
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.IGNORECASE)


def resolve(text: str, names: Iterable[str], project: str | None = None) -> Resolution:
    """Which project `text` names, by whole word, or the one stated.

    `project` given skips the matching entirely: the caller knows, and the
    record says the caller said so. A blank is not a statement. Otherwise
    every roster name is tried as a whole word, case ignored and hyphens as
    whitespace, and exactly one hit resolves.
    """
    if project is not None and project.strip():
        return Resolution(project=project.strip(), candidates=(), rule=RULE_STATED)
    known = tuple(names)
    if not known:
        return Resolution(project=None, candidates=(), rule=RULE_NO_ROSTER)
    named = tuple(name for name in known if _mention(name).search(text))
    if len(named) == 1:
        return Resolution(project=named[0], candidates=named, rule=RULE_ONE)
    if not named:
        return Resolution(project=None, candidates=(), rule=RULE_NONE)
    return Resolution(project=None, candidates=named, rule=RULE_SEVERAL)


def roster_names(corpus: Path = CORPUS) -> tuple[str, ...]:
    """The repository names the corpus's workspace declares, in its order.

    Empty when the submodule is not checked out: an instruction is then
    recorded `unresolved` with `RULE_NO_ROSTER`, distinct from "no project
    named".
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
