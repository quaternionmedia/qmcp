"""Pulling threads from an assistant, as units of work rather than transcripts.

`base` holds the contract every source implements. The sources themselves are
stubs: they declare what they would cost and refuse to pretend they can do it.

**No module here calls a paid service today.** When one does, it goes through
`qmcp.spend`, and `governance/qm/records/DRAFT-no-unattended-spending.md` is why.
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
