"""Pulling threads from an assistant, as units of work rather than transcripts.

`base` holds the contract every source implements. The sources read local
exports and the local session store; `docs/threads.md` describes the archive.

No module here calls a paid service. A source that does spends through
`qmcp.spend`, under `governance/qm/records/DRAFT-no-unattended-spending.md`.
"""

from qmcp.threads.base import (
    Decision,
    Survey,
    Thread,
    ThreadSource,
    Turn,
    to_delta,
)

__all__ = ["Decision", "Survey", "Thread", "ThreadSource", "Turn", "to_delta"]
