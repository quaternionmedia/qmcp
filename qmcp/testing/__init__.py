"""What an offline check stands a qmcp up with: its own server, its own settings.

Two kinds of caller need the same guarantees: the suite's tests, and the
product's own offline checks, which run on a person's machine beside their
real queue. Neither may write into the configured database, bind the
configured port, or read a setting that a shell or a `.env` happens to carry.
`docs/testing.md` is the reference.
"""

from qmcp.testing.server import throwaway_server
from qmcp.testing.settings import isolate_settings

__all__ = ["isolate_settings", "throwaway_server"]
