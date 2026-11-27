"""The apothecary tools: configuration, each tool's request and answer, and how
a refusal reaches the caller.

apothecary is a stand-in behind `httpx.MockTransport`; nothing here needs a
running apothecary. What it answers is `tests/apothecary_answers.json`: an
apothecary's own answers, read from one serving the commit that file names.
`parts` is four of its parts, `site` one site without the keys
`site_left_out` names, which hold its drawing. `jobs` is the empty list that
apothecary answered; `JOB` below follows its `Job` model, apothecary/jobs.py.

Each test names the mutation it was seen to fail under.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import qmcp.config
import qmcp.tools.apothecary as apothecary
from qmcp.tools.apothecary import (
    ApothecaryClient,
    ApothecaryConfig,
    ApothecaryError,
    ApothecaryNotConfiguredError,
)

URL = "http://apothecary.test"
ANSWERS = json.loads(
    Path(__file__).with_name("apothecary_answers.json").read_text(encoding="utf-8"))

JOB = {
    "id": "20260101T001000.000-COM7",
    "kind": "print",
    "machine": {"kind": "printer", "port": "COM7", "identity": None, "path": "printer_1"},
    "site": "garage",
    "part": {"path": "v_slot", "name": "v_slot", "picture": None, "camera": None,
             "registered": True},
    "input": {"name": "v_slot.gcode", "size": 1200, "sha256": None, "file_id": None},
    "started_at": "2026-01-01T00:10:00Z",
    "finished_at": None,
    "outcome": "running",
    "reason": None,
    "progress": 0.25,
    "detail": {"sent": 30, "total": 120},
    "log": [f"line {n}" for n in range(60)],
}


class FakeApothecary:
    """A stand-in apothecary: answers each route it is given, and FastAPI's 404
    for any other."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, httpx.Response] = {}

    def route(self, path: str, body: object, *, status: int = 200,
              headers: dict[str, str] | None = None) -> None:
        self.routes[path] = httpx.Response(status, json=body, headers=headers)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(answer.status_code, content=answer.content,
                              headers=answer.headers)


@pytest.fixture
def fake() -> FakeApothecary:
    return FakeApothecary()


@pytest.fixture
def tools(fake, monkeypatch) -> FakeApothecary:
    """The tools, pointed at the stand-in through `qmcp.tools.apothecary.client`."""
    connected = ApothecaryClient(ApothecaryConfig(url=URL),
                                 transport=httpx.MockTransport(fake.handle))
    monkeypatch.setattr(apothecary, "client", lambda: connected)
    return fake


def settings(**fields) -> SimpleNamespace:
    return SimpleNamespace(**fields)


# --- configuration -----------------------------------------------------------


def test_no_url_is_not_configured_and_says_which_setting():
    """Mutation: fall back to a default address when none is set -- red."""
    with pytest.raises(ApothecaryNotConfiguredError, match="QMCP_APOTHECARY_URL"):
        ApothecaryConfig.from_settings(settings(apothecary_url=None))
    with pytest.raises(ApothecaryNotConfiguredError):
        ApothecaryConfig.from_settings(settings(database_url="sqlite://"))


def test_the_settings_are_read_from_qmcp_environment_names(monkeypatch):
    """Mutation: keep the trailing slash -- red."""
    monkeypatch.setenv("QMCP_APOTHECARY_URL", URL + "/")
    monkeypatch.setenv("QMCP_APOTHECARY_TIMEOUT", "4.5")
    read = ApothecaryConfig.from_settings(qmcp.config.Settings(_env_file=None))
    assert (read.url, read.timeout) == (URL, 4.5)


def test_the_client_is_kept_while_the_settings_name_the_same_apothecary(monkeypatch):
    """Mutation: build a client on every call -- red."""
    current = settings(apothecary_url=URL)
    monkeypatch.setattr(qmcp.config, "get_settings", lambda: current)
    monkeypatch.setattr(apothecary, "_current", None)
    first = apothecary.client()
    assert apothecary.client() is first
    current.apothecary_url = "http://elsewhere.test"
    second = apothecary.client()
    assert second is not first and second.config.url == "http://elsewhere.test"
    assert first._http.is_closed


def test_a_tool_says_apothecary_is_not_configured(monkeypatch):
    monkeypatch.setattr(qmcp.config, "get_settings", lambda: settings(apothecary_url=""))
    monkeypatch.setattr(apothecary, "_current", None)
    with pytest.raises(ApothecaryNotConfiguredError, match="not configured"):
        apothecary.apothecary_parts({})


# --- parts -------------------------------------------------------------------


def test_parts_are_listed_as_rows_without_their_files_or_geometry(tools):
    """Mutation: return apothecary's list as it is -- red, a row carries `files`."""
    tools.route("/parts", ANSWERS["parts"])
    found = apothecary.apothecary_parts({})
    assert [p["name"] for p in found["parts"]] == [p["name"] for p in ANSWERS["parts"]]
    assert all(set(p) == {"name", "description", "category", "tags"} for p in found["parts"])
    assert found["parts"][0]["description"] == ANSWERS["parts"][0]["description"]
    assert found["not_loaded"] == []


def test_the_parts_apothecary_could_not_load_are_named(tools):
    """Mutation: do not read the header -- red."""
    tools.route("/parts", ANSWERS["parts"],
                headers={"X-Parts-Not-Loaded": ANSWERS["parts_not_loaded"] + ", another"})
    assert apothecary.apothecary_parts({})["not_loaded"] == [ANSWERS["parts_not_loaded"],
                                                             "another"]


def test_a_search_finds_a_word_in_a_name_a_category_a_tag_or_a_description(tools):
    """Mutation: leave the category out of what is searched -- red, `hardware`
    is in no name, tag or description here."""
    tools.route("/parts", ANSWERS["parts"])

    def found(search):
        return [p["name"] for p in apothecary.apothecary_parts({"search": search})["parts"]]

    assert found("hose_adapter") == ["hose_adapter"]            # its name
    assert found("Hardware") == ["v_slot"]                      # its category, any case
    assert found("rail") == ["v_slot"]                          # a tag
    assert found("port") == ["hose_adapter"]                    # its description alone
    assert len(found("hose")) < len(ANSWERS["parts"])


def test_a_search_of_several_words_finds_the_parts_that_carry_any(tools):
    """Mutation: ask for every word -- red, no part carries `collection`."""
    tools.route("/parts", ANSWERS["parts"])
    found = apothecary.apothecary_parts({"search": "dust collection"})
    assert [p["name"] for p in found["parts"]] == ["hose_adapter"]
    assert "categories" not in found


def test_a_search_that_finds_nothing_says_which_categories_there_are(tools):
    """Mutation: return the empty list alone -- red."""
    tools.route("/parts", ANSWERS["parts"])
    found = apothecary.apothecary_parts({"search": "flux capacitor"})
    assert found["parts"] == []
    assert found["categories"] == sorted({p["category"] for p in ANSWERS["parts"]})
    assert "categories" not in apothecary.apothecary_parts({})


def test_one_part_carries_its_size_its_parameters_and_what_is_bought(tools):
    """Mutation: read the size from `geometry.size` -- red, it is under `bounds`."""
    part = ANSWERS["part"]
    tools.route(f"/parts/{part['name']}", part)
    found = apothecary.apothecary_part({"name": part["name"]})
    assert found["size_mm"] == part["geometry"]["bounds"]["size"]
    assert found["params"] == part["params"]
    assert found["bought"] == part["bought"]
    assert found["can_generate_stl"] is part["files"]["stl"]["can_generate"]
    assert not {"files", "geometry", "include", "download_url"} & set(found)


def test_a_part_apothecary_measured_no_bounds_for_has_no_size(tools):
    tools.route("/parts/bare", {"name": "bare", "geometry": None, "files": {}})
    found = apothecary.apothecary_part({"name": "bare"})
    assert found["size_mm"] is None and found["can_generate_stl"] is None
    assert "params" not in found


def test_a_part_name_is_one_path_segment(tools):
    """Mutation: put the name in the path unescaped -- red, the path gains a segment."""
    tools.route("/parts/a/b?c", {"name": "a/b?c"})
    apothecary.apothecary_part({"name": "a/b?c"})
    assert tools.requests[-1].url.raw_path.decode() == "/parts/a%2Fb%3Fc"


def test_a_part_needs_a_name(tools):
    with pytest.raises(ValueError, match="name is required"):
        apothecary.apothecary_part({})
    assert tools.requests == []


def test_a_part_apothecary_does_not_hold_is_refused_in_its_words(tools):
    """Mutation: return None for a 404 -- red."""
    missing = ANSWERS["part_missing"]
    tools.route("/parts/no_such_part", missing["body"], status=missing["status"])
    with pytest.raises(ApothecaryError, match="no_such_part") as refused:
        apothecary.apothecary_part({"name": "no_such_part"})
    assert refused.value.status == 404
    assert refused.value.detail == missing["body"]["detail"]


# --- sites -------------------------------------------------------------------


def test_sites_are_listed_by_name(tools):
    tools.route("/sites", ANSWERS["sites"])
    assert apothecary.apothecary_sites({}) == ANSWERS["sites"]


def test_a_site_is_its_structures_without_its_drawing(tools):
    """Mutation: return apothecary's site as it is -- red, it carries `tree` and `scad`."""
    site = dict(ANSWERS["site"], tree={"children": ["x"] * 500}, scad="cube(1);" * 500)
    tools.route(f"/sites/{site['name']}", site)
    found = apothecary.apothecary_site({"name": site["name"]})
    assert set(found) == {"name", "is_valid", "violations", "structures"}
    assert found["is_valid"] is site["is_valid"]
    assert [s["name"] for s in found["structures"]] == [s["name"] for s in site["structures"]]
    printer = next(s for s in site["structures"] if s["build_volume"])
    shown = next(s for s in found["structures"] if s["name"] == printer["name"])
    assert shown == {
        "name": printer["name"], "material": printer["material"],
        "status": printer["status"], "build_volume": printer["build_volume"],
        "systems": [s["name"] for s in printer["substructures"]],
    }


# --- jobs --------------------------------------------------------------------


def test_no_jobs_is_an_empty_list(tools):
    tools.route("/jobs", ANSWERS["jobs"])
    assert apothecary.apothecary_jobs({}) == []


def test_jobs_are_listed_without_what_the_machine_said(tools):
    """Mutation: return each job as it is -- red, a row carries `log`."""
    tools.route("/jobs", [JOB])
    found = apothecary.apothecary_jobs({})
    assert len(found) == 1 and "log" not in found[0] and "detail" not in found[0]
    assert (found[0]["id"], found[0]["outcome"], found[0]["progress"]) == (
        JOB["id"], "running", 0.25)
    assert found[0]["machine"] == JOB["machine"]


def test_a_site_or_a_kind_is_sent_to_apothecary(tools):
    """Mutation: send no query -- red."""
    tools.route("/jobs", [])
    apothecary.apothecary_jobs({"site": "garage", "kind": "print"})
    assert dict(tools.requests[-1].url.params) == {"site": "garage", "kind": "print"}
    apothecary.apothecary_jobs({})
    assert dict(tools.requests[-1].url.params) == {}


def test_one_job_carries_the_tail_of_what_its_machine_said(tools):
    """Mutation: return the whole log -- red, it is longer than LOG_LINES."""
    tools.route(f"/jobs/{JOB['id']}", JOB)
    found = apothecary.apothecary_job({"id": JOB["id"]})
    assert found["log"] == JOB["log"][-apothecary.LOG_LINES:]
    assert len(JOB["log"]) > apothecary.LOG_LINES
    assert found["id"] == JOB["id"]


def test_a_job_apothecary_does_not_hold_is_refused_in_its_words(tools):
    missing = ANSWERS["job_missing"]
    tools.route("/jobs/no-such-job", missing["body"], status=missing["status"])
    with pytest.raises(ApothecaryError, match=missing["body"]["detail"]):
        apothecary.apothecary_job({"id": "no-such-job"})


# --- health, and what is not apothecary ----------------------------------------


def test_health_is_apothecarys_own_answer(tools):
    tools.route("/health", ANSWERS["health"])
    assert apothecary.apothecary_health({}) == ANSWERS["health"]


def test_an_answer_of_the_wrong_shape_is_an_error_not_a_result(tools):
    """Mutation: drop the shape check -- red, the tool raises AttributeError or returns."""
    tools.route("/parts", {"detail": "this is not a list"})
    with pytest.raises(ApothecaryError, match="not the list"):
        apothecary.apothecary_parts({})


def test_a_body_that_is_not_json_is_an_error():
    def page(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>a page</html>",
                              headers={"content-type": "text/html"})

    client = ApothecaryClient(ApothecaryConfig(url=URL), transport=httpx.MockTransport(page))
    with pytest.raises(ApothecaryError, match="text/html.*, not JSON"):
        client.get("/health")


def test_an_apothecary_that_does_not_answer_is_named_with_its_address():
    """Mutation: let httpx's own error through -- red."""
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    down = ApothecaryClient(ApothecaryConfig(url=URL), transport=httpx.MockTransport(refuse))
    with pytest.raises(ApothecaryError, match=f"{URL} could not be reached"):
        down.get("/health")
    slow = ApothecaryClient(ApothecaryConfig(url=URL, timeout=2),
                            transport=httpx.MockTransport(stall))
    with pytest.raises(ApothecaryError, match=r"did not answer GET /health in time \(.*2 s\)"):
        slow.get("/health")


def test_every_apothecary_tool_is_published():
    """Mutation: leave the module out of `published_registry` -- red.

    Asked in a process of its own: this file imports the module, which
    registers its tools whatever `published_registry` imports.
    """
    asked = subprocess.run(
        [sys.executable, "-c",
         "from qmcp.tools import published_registry\n"
         "print('\\n'.join(t.name for t in published_registry().list_tools()))"],
        capture_output=True, text=True, timeout=120)
    assert asked.returncode == 0, asked.stderr
    names = {name for name in asked.stdout.split() if name.startswith("apothecary_")}
    assert names == {"apothecary_parts", "apothecary_part", "apothecary_sites",
                     "apothecary_site", "apothecary_jobs", "apothecary_job",
                     "apothecary_health"}


def test_the_client_sends_nothing_but_gets(tools):
    """Every tool, run once: each request the stand-in saw is a GET."""
    tools.route("/parts", ANSWERS["parts"])
    tools.route("/parts/v_slot", ANSWERS["part"])
    tools.route("/sites", ANSWERS["sites"])
    tools.route("/sites/garage", ANSWERS["site"])
    tools.route("/jobs", [JOB])
    tools.route(f"/jobs/{JOB['id']}", JOB)
    tools.route("/health", ANSWERS["health"])
    apothecary.apothecary_parts({})
    apothecary.apothecary_part({"name": "v_slot"})
    apothecary.apothecary_sites({})
    apothecary.apothecary_site({"name": "garage"})
    apothecary.apothecary_jobs({})
    apothecary.apothecary_job({"id": JOB["id"]})
    apothecary.apothecary_health({})
    assert len(tools.requests) == 7
    assert {request.method for request in tools.requests} == {"GET"}
