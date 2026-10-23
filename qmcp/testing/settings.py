"""Settings a check reads are the check's, not the machine's.

`qmcp.config.Settings` reads every `QMCP_*` variable in the environment and a
`.env` in the working directory, and `get_settings` keeps the first answer for
the life of the process. Left alone, a developer's shell or `.env` -- a
`QMCP_CONVERSE_RUNTIME`, a `QMCP_DATABASE_URL` -- decides what a test sees, so
a test passes on one machine and fails on the next for a reason neither shows.
"""

from __future__ import annotations

import os
from typing import Any

from qmcp.config import Settings, get_settings


def isolate_settings(monkeypatch: Any) -> None:
    """Take every `QMCP_*` variable and the `.env` file out of what settings read.

    `monkeypatch` is pytest's, or anything with its `delenv` and `setitem`;
    undoing it puts the variables back and reads `.env` again. The settings
    cache is cleared here, so the next reader builds from what is left.
    Pydantic matches the prefix without regard to case, so this does too.
    """
    for name in [name for name in os.environ if name.upper().startswith("QMCP_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
