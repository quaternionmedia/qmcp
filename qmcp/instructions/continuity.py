"""Continuity comes from qmcp, not the model: a project's history, read from the record.

What is carried forward is what ran. `history` reads a project's earlier
instructions that ran -- `done` or `failed`, with the runtime and what it
said -- oldest first, and the brief carries them to whichever runtime is
chosen. A held or unanswered instruction is not carried. Because the memory is
the record, the runtime can change between two instructions and the second
still knows what the first found.

`last_clone` is the directory the project's most recent act ran in, so a
`--cwd` given once serves every later instruction in that project.

`qmcp.instructions.act` writes the ids of the turns it handed over into the
row's `detail`.

What this cannot do: know which earlier instructions are relevant. It carries
the most recent `LIMIT` that ran, in order; older history is in the record,
not in the brief.
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
