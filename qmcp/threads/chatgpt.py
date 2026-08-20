"""ChatGPT threads. A STUB: it establishes nothing and calls nothing.

WHAT IS DECIDED AND WHAT IS NOT. The shape is decided -- this is a
`ThreadSource`, its `survey` is free, its `fetch` spends against a budget a
person issued, and its deltas name a perspective. What is not decided is where
the threads come from, and that is why every method below refuses rather than
guessing.

THE OPEN QUESTION, STATED PLAINLY. There are two ways in and they have
different costs and different records behind them:

  an export file   A conversation export the operator downloaded. Reading it is
                   free, so `survey` could answer properly and `fetch` would
                   never spend at all. It is also a snapshot: it is as current
                   as the day it was exported.

  a live API       Current, and metered. `survey` could not answer for free if
                   listing is itself billed, so it would return `unknown` with
                   that as the reason, and a person would decide whether to
                   spend to find out how much there is to spend on.

**The second needs a credential, and this repository holds none.** Choosing is a
person's, and until they choose, a stub that says so is more use than one that
picks.

ONE DIFFERENCE FROM THE CLAUDE SOURCE, AND IT IS NOT COSMETIC. Its perspective
is its own -- `chatgpt/thread`. Two assistants discussing the same work produce
two sets of deltas, and neither is the other's duplicate: they are two
perspectives on one strand. `same-as` is how somebody says they are the same
strand once they have read both, and
`governance/qm/records/DRAFT-deltas-compose.md` 4 is why neither address is
retired when they do.

NOTHING HERE IS SCHEDULED AND NOTHING RETRIES. When this is implemented, every
call goes through the `Budget` it was handed, checked before the call rather
than after -- `governance/qm/records/DRAFT-no-unattended-spending.md`.
"""

from __future__ import annotations

from qmcp.spend import Budget, unknown
from qmcp.threads.base import Decision, Survey, Thread, ThreadSource

NOT_BUILT = (
    "the ChatGPT thread source is a stub. Neither route is chosen: an export "
    "file (free to read, a snapshot) or the live API (current, metered, and "
    "needing a credential this repository does not hold)."
)


class ChatGPTThreads(ThreadSource):
    """Threads from ChatGPT. Implements the contract and does none of the work."""

    name = "chatgpt"

    # A default that says what this source is rather than what it contains. It
    # is a claim about level: this source speaks about whole conversations and
    # what they settled, not about turns.
    perspective = "chatgpt/thread"

    def survey(self) -> Survey:
        """Establishes nothing, and says why rather than returning zero.

        `available=0` would claim there are no threads. Nobody looked, and the
        difference between those is the whole reason `unknown` carries a
        reason.
        """
        return Survey(
            source=self.name,
            available=unknown(NOT_BUILT),
            would_need=unknown(NOT_BUILT),
            note="Nothing was spent reaching this answer.",
        )

    def fetch(self, ids: list[str], budget: Budget) -> list[Thread]:
        raise NotImplementedError(NOT_BUILT)

    def decisions(self, thread: Thread, budget: Budget) -> list[Decision]:
        raise NotImplementedError(NOT_BUILT)
