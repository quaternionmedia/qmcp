"""Entry point for `python -m qmcp`, and for `python qmcp` run from the checkout.

`uv run qmcp <command>` is the declared form. The other two start the same CLI.

Run as `python qmcp` -- the directory, not the module -- Python puts this
package's own directory first on `sys.path`, where each of its modules would
shadow a top-level one of the same name. It did: `qmcp/logging.py` replaced the
standard library's `logging` for everything imported afterwards, and the
command died inside `uvicorn` before printing a line. So the project root goes
first instead, before anything else is imported, the way `ci/cli.py` in the
governance corpus does for the same reason.
"""

import os
import sys

if not __package__:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
    sys.path.insert(0, os.path.dirname(_here))

from qmcp.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
