"""A pending request is one somebody can still answer, and "oldest" is oldest.

Every reader that takes "the first pending request" -- `qmcp human voice`
without an id, `VoiceApprovalLoop.run_forever` -- asks a person about it. A
request past its expiry cannot be answered, and a listing that offered it
put a months-old test fixture in front of the person first. The listing was
also newest first while every page and help text said the oldest was
answered.
"""

from datetime import UTC, datetime, timedelta

from click.testing import CliRunner
from sqlmodel import Session, SQLModel, create_engine

import qmcp.server
from qmcp.cli import cli
from qmcp.db.models import HumanRequest, HumanResponse


def _later(hours: float):
    """A `datetime` whose `now` runs `hours` ahead, for the server's clock."""

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(hours=hours)

    return Later


def _create(client, request_id: str, timeout_seconds: int) -> None:
    response = client.post(
        "/v1/human/requests",
        json={
            "id": request_id,
            "request_type": "approval",
            "prompt": f"{request_id}?",
            "options": ["approve", "hold"],
            "timeout_seconds": timeout_seconds,
        },
    )
    assert response.status_code == 201, response.text


def _pending_ids(client, **params) -> list[str]:
    response = client.get("/v1/human/requests", params={"status": "pending", **params})
    assert response.status_code == 200, response.text
    return [r["id"] for r in response.json()["requests"]]


# --- the server's listing ------------------------------------------------------


def test_a_request_past_its_expiry_is_not_listed_as_pending(client, monkeypatch):
    _create(client, "short-lived", timeout_seconds=60)
    _create(client, "day-long", timeout_seconds=86400)
    assert set(_pending_ids(client)) == {"short-lived", "day-long"}

    monkeypatch.setattr(qmcp.server, "datetime", _later(hours=2))

    assert _pending_ids(client) == ["day-long"]


def test_leaving_it_out_changes_nothing_stored(client, monkeypatch):
    """Listing stays free of side effects, which is why the voice loop polls
    it rather than `get_human_request`."""
    _create(client, "short-lived", timeout_seconds=60)
    monkeypatch.setattr(qmcp.server, "datetime", _later(hours=2))

    assert _pending_ids(client) == []
    everything = client.get("/v1/human/requests").json()["requests"]
    assert [(r["id"], r["status"]) for r in everything] == [("short-lived", "pending")]


def test_oldest_first_is_oldest_first(client):
    for request_id in ("first", "second", "third"):
        _create(client, request_id, timeout_seconds=3600)

    assert _pending_ids(client) == ["third", "second", "first"]
    assert _pending_ids(client, oldest_first="true") == ["first", "second", "third"]


def test_the_client_asks_the_server_for_oldest_first(client):
    """The real client, not a copy of its request-building: `test_client.py`'s
    fixture re-implements the method, so a parameter the real one forgot to
    send would pass there."""
    from qmcp.client.mcp_client import MCPClient

    for request_id in ("first", "second"):
        _create(client, request_id, timeout_seconds=3600)
    real = MCPClient()
    real._client.close()
    real._client = client  # a TestClient is an httpx.Client

    listed = real.list_human_requests(status_filter="pending", oldest_first=True)
    assert [r.id for r in listed] == ["first", "second"]


# --- `qmcp human list`, which reads the database directly ----------------------


def _database(tmp_path):
    path = tmp_path / "qmcp.db"
    engine = create_engine(f"sqlite:///{path.as_posix()}")
    SQLModel.metadata.create_all(engine)
    now = datetime.now(UTC).replace(tzinfo=None)
    with Session(engine) as session:
        session.add(HumanRequest(
            id="stale-fixture", request_type="approval", prompt="From January?",
            created_at=now - timedelta(days=250), expires_at=now - timedelta(days=249)))
        session.add(HumanRequest(
            id="live-launch", request_type="approval", prompt="Launch the audit?",
            options=["approve", "hold"],
            created_at=now - timedelta(hours=1), expires_at=now + timedelta(hours=23)))
        session.add(HumanRequest(
            id="answered", request_type="approval", prompt="Already done?",
            created_at=now - timedelta(hours=2), expires_at=now - timedelta(hours=1)))
        session.add(HumanResponse(request_id="answered", response="approve"))
        session.commit()
    return path


def test_human_list_shows_only_what_can_still_be_answered(tmp_path):
    result = CliRunner().invoke(cli, ["human", "list", "--database", str(_database(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "live-launch" in result.output
    assert "stale-fixture" not in result.output
    assert "answered" not in result.output.replace("--all includes", "")
    assert "1 waiting." in result.output


def test_human_list_all_marks_an_expired_request_as_expired(tmp_path):
    result = CliRunner().invoke(
        cli, ["human", "list", "--all", "--database", str(_database(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "[x] stale-fixture" in result.output
    assert "expired:" in result.output
    assert "[?] live-launch" in result.output
    # Answered after its expiry passed is still answered, not expired.
    assert "[=] answered" in result.output
