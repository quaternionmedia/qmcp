"""Tools that read apothecary, the workshop's parts and machines, over its HTTP API.

`apothecary_parts`, `apothecary_part`, `apothecary_sites`, `apothecary_site`,
`apothecary_jobs`, `apothecary_job` and `apothecary_health` each ask the
apothecary named by `QMCP_APOTHECARY_URL` and return what it answered, cut
down to what a caller that is not apothecary's own page reads: a part without
its file links and its colour, a site without its drawing.
`docs/integrations/apothecary.md` is the reference: the setting, each tool's
input and output, and the errors.

Every tool here reads. None starts a machine, changes a part or moves a
structure.

Nothing from apothecary is imported; every call is one HTTP GET. apothecary
answers this machine only and asks for no account, so the tools send none. A
request apothecary refuses raises `ApothecaryError` carrying apothecary's own
`detail`, which the server records as the invocation's error. Between calls
the tools keep one thing: the HTTP client.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from qmcp import config
from qmcp.tools.registry import tool_registry

# Seconds apothecary has to answer one request when `QMCP_APOTHECARY_TIMEOUT`
# is unset (the setting's own default), and the most a connection may take to
# open.
DEFAULT_TIMEOUT: float = config.Settings.model_fields["apothecary_timeout"].default
CONNECT_TIMEOUT = 3.0

NOT_CONFIGURED = (
    "apothecary is not configured: set QMCP_APOTHECARY_URL to its server's "
    "address, such as http://127.0.0.1:8000"
)

# The header apothecary names the parts it could not load in, on `GET /parts`.
NOT_LOADED_HEADER = "X-Parts-Not-Loaded"

# What a part's row carries in a list, and what one part adds to that.
PART_ROW = ("name", "description", "category", "tags")
PART_FACTS = ("kinematics", "motion", "bought", "has_params", "params", "readme", "source_file")

# What a structure of a site carries, and a job in a list.
STRUCTURE_FACTS = ("name", "material", "status", "build_volume")
JOB_FACTS = ("id", "kind", "machine", "site", "part", "input", "started_at", "finished_at",
             "outcome", "reason", "progress")

# The most lines of a machine's own words `apothecary_job` returns.
LOG_LINES = 40


class ApothecaryError(RuntimeError):
    """apothecary could not be asked, or refused what it was asked.

    `status` is apothecary's HTTP status when it answered, and `detail` its
    `detail`.
    """

    def __init__(self, message: str, *, status: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class ApothecaryNotConfiguredError(ApothecaryError):
    """The settings name no apothecary."""


# ---------------------------------------------------------------------------
# Configuration and the client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApothecaryConfig:
    """Where apothecary answers, and how long it has."""

    url: str
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_settings(cls, settings: Any = None) -> ApothecaryConfig:
        """The configuration qmcp's settings hold.

        Raises `ApothecaryNotConfiguredError` for no URL. Fields are read with
        defaults, so a settings object without them is an unconfigured
        apothecary.
        """
        settings = settings if settings is not None else config.get_settings()
        url = (getattr(settings, "apothecary_url", None) or "").strip().rstrip("/")
        if not url:
            raise ApothecaryNotConfiguredError(NOT_CONFIGURED)
        timeout = getattr(settings, "apothecary_timeout", None)
        return cls(url=url, timeout=DEFAULT_TIMEOUT if timeout is None else float(timeout))


class ApothecaryClient:
    """One apothecary, read."""

    def __init__(self, cfg: ApothecaryConfig, *,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.config = cfg
        self._http = httpx.Client(
            base_url=cfg.url,
            timeout=httpx.Timeout(cfg.timeout, connect=min(CONNECT_TIMEOUT, cfg.timeout)),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """The JSON apothecary answers `GET path` with."""
        return self.read(path, params)[0]

    def read(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, httpx.Headers]:
        """The JSON apothecary answers `GET path` with, and the answer's headers."""
        try:
            response = self._http.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise ApothecaryError(
                f"apothecary at {self.config.url} did not answer GET {path} in time "
                f"({type(exc).__name__}, {self.config.timeout:g} s)"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApothecaryError(
                f"apothecary at {self.config.url} could not be reached for GET {path}: {exc}"
            ) from exc
        if response.is_error:
            detail = _detail(response)
            raise ApothecaryError(
                f"apothecary refused GET {path} ({response.status_code}): {_text(detail)}",
                status=response.status_code, detail=detail,
            )
        try:
            return response.json(), response.headers
        except ValueError:
            kind = response.headers.get("content-type", "a body")
            raise ApothecaryError(
                f"apothecary answered GET {path} with {kind}, not JSON",
                status=response.status_code,
            ) from None


def _detail(response: httpx.Response) -> Any:
    """apothecary's `detail` for a refusal, or the body when it carries none."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:300] or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


def _text(detail: Any) -> str:
    """A `detail` as one line: a string as it is, a validation list as `where: what`."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail and all(isinstance(e, dict) for e in detail):
        said = []
        for error in detail:
            where = ".".join(str(step) for step in error.get("loc") or ())
            what = str(error.get("msg", error))
            said.append(f"{where}: {what}" if where else what)
        return "; ".join(said)
    return str(detail)


_current: ApothecaryClient | None = None
_current_lock = threading.Lock()


def client() -> ApothecaryClient:
    """The client for the apothecary qmcp's settings name, kept while they name the same one.

    Raises `ApothecaryNotConfiguredError` when they name none.
    """
    global _current
    cfg = ApothecaryConfig.from_settings()
    with _current_lock:
        if _current is None or _current.config != cfg:
            if _current is not None:
                _current.close()
            _current = ApothecaryClient(cfg)
        return _current


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def _expect(value: Any, kind: type, path: str) -> Any:
    if not isinstance(value, kind):
        raise ApothecaryError(
            f"apothecary answered GET {path} with {type(value).__name__}, "
            f"not the {kind.__name__} apothecary serves there"
        )
    return value


def _segment(value: Any) -> str:
    """A name as one path segment: `/` and every other reserved character escaped."""
    return quote(str(value), safe="")


def _required(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required")
    return value


def part_row(part: dict[str, Any]) -> dict[str, Any]:
    """A part as a list carries it: its name, what it is, its category and its tags."""
    return {key: part.get(key) for key in PART_ROW}


def words_of(search: Any) -> list[str]:
    """The words a search is made of, lowercased: runs of letters and digits."""
    return re.findall(r"[a-z0-9]+", str(search or "").lower())


def part_matches(row: dict[str, Any], words: list[str]) -> bool:
    """Whether a part's row carries any of the words, in its name, its category,
    its tags or its description, whatever their case."""
    held = " ".join([str(row.get("name") or ""), str(row.get("category") or ""),
                     " ".join(str(tag) for tag in row.get("tags") or []),
                     str(row.get("description") or "")]).lower()
    return any(word in held for word in words)


def part_view(part: dict[str, Any]) -> dict[str, Any]:
    """One part: its row, its size in millimetres, and the facts apothecary holds on it.

    `size_mm` is the size of its bounds, null for a part apothecary measured
    none for. `can_generate_stl` is whether apothecary can make its mesh, with
    `stl_note` saying why not. A fact apothecary did not send is left out.
    """
    view = part_row(part)
    bounds = ((part.get("geometry") or {}).get("bounds") or {})
    view["size_mm"] = bounds.get("size")
    stl = ((part.get("files") or {}).get("stl") or {})
    view["can_generate_stl"] = stl.get("can_generate")
    view["stl_note"] = stl.get("note")
    for key in PART_FACTS:
        if key in part:
            view[key] = part[key]
    return view


def site_view(site: dict[str, Any]) -> dict[str, Any]:
    """A site without its drawing: whether it is valid, what breaks its rules, and
    each structure's name, material, status, build volume and the systems it has."""
    structures = []
    for structure in site.get("structures") or []:
        row = {key: structure.get(key) for key in STRUCTURE_FACTS}
        row["systems"] = [s.get("name") for s in structure.get("substructures") or []]
        structures.append(row)
    return {
        "name": site.get("name"),
        "is_valid": site.get("is_valid"),
        "violations": site.get("violations") or [],
        "structures": structures,
    }


def job_row(job: dict[str, Any]) -> dict[str, Any]:
    """A job as a list carries it, without the machine's own words."""
    return {key: job.get(key) for key in JOB_FACTS}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_registry.register(
    name="apothecary_parts",
    description=(
        "List the parts in the workshop's parts library: each one's name, what it is, "
        "its category and its tags. `search` narrows the list to the parts that carry any "
        "of its words in their name, category, tags or description; a search that finds "
        "none also returns `categories`, the categories there are. `not_loaded` names the "
        "parts apothecary holds and could not load."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "search": {"type": "string",
                       "description": "One or more words to look for, such as dust or hose"},
        },
    },
)
def apothecary_parts(params: dict[str, Any]) -> dict[str, Any]:
    listed, headers = client().read("/parts")
    parts = [part_row(p) for p in _expect(listed, list, "/parts") if isinstance(p, dict)]
    named = headers.get(NOT_LOADED_HEADER, "")
    found: dict[str, Any] = {
        "parts": parts,
        "not_loaded": [name.strip() for name in named.split(",") if name.strip()],
    }
    words = words_of(params.get("search"))
    if words:
        found["parts"] = [p for p in parts if part_matches(p, words)]
        if not found["parts"]:
            found["categories"] = sorted({p["category"] for p in parts if p["category"]})
    return found


@tool_registry.register(
    name="apothecary_part",
    description=(
        "One part of the workshop's parts library, by name: what it is, its category and "
        "tags, its size in millimetres, the parameters it takes at their own values, what "
        "it needs that is bought and not made, and whether its mesh can be generated. A "
        "kind of machine or board, such as ender3, is a part."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string",
                     "description": "The part's name, as apothecary_parts lists it"},
        },
        "required": ["name"],
    },
)
def apothecary_part(params: dict[str, Any]) -> dict[str, Any]:
    path = f"/parts/{_segment(_required(params, 'name'))}"
    return part_view(_expect(client().get(path), dict, path))


@tool_registry.register(
    name="apothecary_sites",
    description="List the names of the sites apothecary holds: a workshop, a room, a library.",
    input_schema={"type": "object", "properties": {}},
)
def apothecary_sites(params: dict[str, Any]) -> list[str]:
    return _expect(client().get("/sites"), list, "/sites")


@tool_registry.register(
    name="apothecary_site",
    description=(
        "One site, by name: whether it is valid, what breaks its rules, and each structure "
        "in it -- a bench, a printer, a router -- with its material, its status, its build "
        "volume in millimetres and the systems it has. A site is a place, such as garage; a "
        "machine such as printer_1 is a structure inside a site, never a site."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string",
                     "description": "The site's name, as apothecary_sites lists it"},
        },
        "required": ["name"],
    },
)
def apothecary_site(params: dict[str, Any]) -> dict[str, Any]:
    path = f"/sites/{_segment(_required(params, 'name'))}"
    return site_view(_expect(client().get(path), dict, path))


@tool_registry.register(
    name="apothecary_jobs",
    description=(
        "List the jobs machines ran or are running, the running ones first, then the "
        "newest: each one's id, kind, machine, site, part, file, when it started and "
        "finished, its outcome (running, done, cancelled or failed) and its progress from "
        "0 to 1. `site` or `kind` narrows the list."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "site": {"type": "string", "description": "Only this site's jobs"},
            "kind": {"type": "string", "description": "Only the jobs of this kind, such as print"},
        },
    },
)
def apothecary_jobs(params: dict[str, Any]) -> list[dict[str, Any]]:
    query = {key: params[key] for key in ("site", "kind") if params.get(key)}
    listed = _expect(client().get("/jobs", params=query or None), list, "/jobs")
    return [job_row(job) for job in listed if isinstance(job, dict)]


@tool_registry.register(
    name="apothecary_job",
    description=(
        "One job, by id: what apothecary_jobs lists for it, and the last lines of what "
        "its machine said."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string",
                     "description": "The job's id, as apothecary_jobs lists it"},
        },
        "required": ["id"],
    },
)
def apothecary_job(params: dict[str, Any]) -> dict[str, Any]:
    path = f"/jobs/{_segment(_required(params, 'id'))}"
    job = _expect(client().get(path), dict, path)
    view = job_row(job)
    log = job.get("log")
    view["log"] = log[-LOG_LINES:] if isinstance(log, list) else []
    return view


@tool_registry.register(
    name="apothecary_health",
    description="Whether apothecary answers, and the version it says it is.",
    input_schema={"type": "object", "properties": {}},
)
def apothecary_health(params: dict[str, Any]) -> dict[str, Any]:
    return _expect(client().get("/health"), dict, "/health")


__all__ = [
    "ApothecaryClient",
    "ApothecaryConfig",
    "ApothecaryError",
    "ApothecaryNotConfiguredError",
    "client",
    "job_row",
    "part_matches",
    "part_row",
    "part_view",
    "site_view",
    "words_of",
]
