"""Continuity comes from qmcp, not the model: a project's history, read from the record.

**WHAT IS CARRIED FORWARD IS WHAT RAN.** The inbox keeps every instruction a
person recorded, and each one acted on keeps what came back. Before the next
instruction in the same project runs, `history` reads the earlier ones that
ran -- `done` or `failed`, with the runtime that ran them and what it said --
oldest first, and the brief carries them to whichever runtime is chosen. A
held or unanswered instruction is not carried: nothing ran, so there is
nothing it found. The same record that says what happened is the memory the
next run starts from, which is why a runtime can be swapped between two
instructions and the second still knows what the first found.

**THE CLONE IS REMEMBERED THE SAME WAY.** `last_clone` is the directory the
project's most recent act ran in, so a `--cwd` given once serves every later
instruction in that project, and a person never has to say it twice.

**THE RECORD SAYS WHAT WAS CARRIED.** `qmcp.instructions.act` writes the ids
of the turns it handed over into the row's `detail`, so a reader of any
outcome can see exactly which earlier work the runtime was told about.

WHAT THIS CANNOT DO. Know which earlier instructions are relevant. It carries
the most recent `LIMIT` that ran, in order, and nothing else; a project's
history longer than that is in the record, not in the brief.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlmodel import col, select

from qmcp.db.models import Instruction, InstructionStatus
from qmcp.integrations.agents import Turn

# How many earlier instructions a brief carries. Enough to continue a line of
# work; few enough that a small model's window still holds the instruction.
LIMIT = 5

RAN = (InstructionStatus.DONE, InstructionStatus.FAILED)


def _stamp(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def history(rows: Any, project: str | None, before: str, limit: int = LIMIT) -> tuple[Turn, ...]:
    """The project's earlier instructions that ran, oldest first, without `before`.

    `rows` is the inbox (`qmcp.instructions.act.Rows`). A row with no project
    has no history: an unresolved instruction belongs to nothing yet.
    """
    if not project:
        return ()
    with rows() as session:
        found = session.exec(
            select(Instruction)
            .where(Instruction.project == project, col(Instruction.status).in_(RAN),
                   Instruction.id != before)
            .order_by(col(Instruction.acted_at).desc(), col(Instruction.created_at).desc())
            .limit(limit)).all()
        turns = [Turn(id=row.id, instruction=row.text, status=row.status.value,
                      outcome=row.outcome_text or "", runtime=row.runtime,
                      at=_stamp(row.acted_at or row.created_at)) for row in found]
    return tuple(reversed(turns))


def last_clone(rows: Any, project: str | None, before: str) -> Path | None:
    """The directory the project's most recent act ran in, or None."""
    if not project:
        return None
    with rows() as session:
        row = session.exec(
            select(Instruction)
            .where(Instruction.project == project, col(Instruction.cwd).is_not(None),
                   Instruction.id != before)
            .order_by(col(Instruction.acted_at).desc(), col(Instruction.updated_at).desc())
            .limit(1)).first()
        return Path(row.cwd) if row is not None else None


__all__ = ["LIMIT", "history", "last_clone"]
