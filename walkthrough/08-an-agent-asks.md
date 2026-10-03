# 08 — An agent asks

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

**The problem.** `qmcp_mcp.py` is the server a coding assistant connects to.
It listed the human queue and answered it, and could neither put a question on
it nor wait for one — so a session in any repository had no way to put a
question to the person at the speaker, or at `qmcp human respond`, through it.
Three tools close that: `create_human_request`, `await_human_response`, and
`ask_human`, which is the two composed.

The tools are HTTP calls against a qmcp server, so this page starts one: on an
ephemeral port, over a database of its own, because the configured one holds
somebody's queue.

    >>> import tempfile, threading, time
    >>> from contextlib import ExitStack
    >>> from pathlib import Path
    >>> from unittest.mock import patch
    >>> import qmcp_mcp
    >>> from qmcp.integrations.voice.check import throwaway_server

    >>> stack = ExitStack()
    >>> url = stack.enter_context(throwaway_server(Path(tempfile.mkdtemp()) / "queue.db"))

## The question is a row on the queue

A closed choice is an `approval`, and the options are the grammar the voice
loop will speak:

    >>> created = qmcp_mcp.create_human_request(
    ...     "Merge the pin bump?", options=["approve", "hold"],
    ...     request_id="walkthrough-pin", server_url=url)
    >>> created["status"], created["request_type"]
    ('pending', 'approval')

It is on the same queue `qmcp human list` reads, with nothing to mark it as an
agent's rather than a flow's:

    >>> [r["id"] for r in qmcp_mcp.list_human_requests(status_filter="pending", server_url=url)]
    ['walkthrough-pin']

## Waiting reads the listing, and nothing else, until the answer lands

The answer comes from somewhere else — a person speaking to the voice loop, or
typing `qmcp human respond`. Here a second thread stands in for them and
answers a moment from now, while the tool waits. Every route the wait reads is
recorded on the way through, because which routes it reads is the point:

    >>> real_get, routes = qmcp_mcp.httpx.get, []
    >>> def recording_get(request_url, **kwargs):
    ...     routes.append(request_url.rsplit("/v1/", 1)[1])
    ...     return real_get(request_url, **kwargs)

    >>> answer = threading.Timer(
    ...     0.3, qmcp_mcp.submit_human_response,
    ...     args=("walkthrough-pin", "approve"),
    ...     kwargs={"responded_by": "walkthrough", "server_url": url})
    >>> answer.start()
    >>> with patch("qmcp_mcp.httpx.get", side_effect=recording_get):
    ...     result = qmcp_mcp.await_human_response(
    ...         "walkthrough-pin", timeout_seconds=10, poll_seconds=0.05, server_url=url)
    >>> result["status"], result["response"]["response"], result["response"]["responded_by"]
    ('answered', 'approve', 'walkthrough')

`responded_by` is the only trace of how it was answered: the voice loop records
`vox`, a typed answer records whatever the typist gave, and the tool cannot
tell the two apart otherwise.

The listing, for as long as the id was on it, and the request itself once,
after — `AGENTS.md`'s "One read in this API is a write" is why the order
matters:

    >>> set(routes[:-1]), routes[-1]
    ({'human/requests'}, 'human/requests/walkthrough-pin')

## Nobody answers

A wait that runs out reads nothing, and leaves the question where it was:

    >>> _ = qmcp_mcp.create_human_request(
    ...     "Ship tonight?", options=["approve", "hold"],
    ...     request_id="walkthrough-quiet", server_url=url)
    >>> waited = qmcp_mcp.await_human_response("walkthrough-quiet", timeout_seconds=0, server_url=url)
    >>> waited["status"], waited["response"]
    ('timeout', None)
    >>> [r["id"] for r in qmcp_mcp.list_human_requests(status_filter="pending", server_url=url)]
    ['walkthrough-quiet']

The question is still answerable: the agent gave up on it, and nothing about
the queue says so. An answer that arrives later is recorded like any other.

## The call an agent makes

`ask_human` is the two composed, and makes the id. Without options the question
is an `input`, and the answer is whatever was said. The person at the speaker
waits for a question with a made id to appear, rather than for a moment to
pass, so that what they answer is this question and not the one left pending
above:

    >>> def whoever_is_at_the_speaker():
    ...     while True:
    ...         made = [r for r in qmcp_mcp.list_human_requests(status_filter="pending", server_url=url)
    ...                 if r["id"].startswith("ask-")]
    ...         if made:
    ...             break
    ...         time.sleep(0.05)
    ...     qmcp_mcp.submit_human_response(
    ...         made[0]["id"], "the pin, not the lock", responded_by="walkthrough", server_url=url)
    >>> threading.Thread(target=whoever_is_at_the_speaker).start()
    >>> asked = qmcp_mcp.ask_human(
    ...     "Which file changed?", timeout_seconds=60, poll_seconds=0.05, server_url=url)
    >>> asked["status"], asked["response"]["response"]
    ('answered', 'the pin, not the lock')
    >>> asked["request_id"].startswith("ask-")
    True

    >>> stack.close()

## What this page does not claim

That a question expires. The server holds the floor on how long a question
stays answerable, and this page does not wait that long; the expired path is
held by `tests/test_qmcp_mcp.py` against a scripted listing. Nor that anyone
heard the question: the server is spoken to over HTTP here, and whether the
voice loop speaks it is `qmcp cookbook voice`'s claim, not this page's.

The page has been seen to fail. With the wait reading the listing without its
`status=pending` filter, the answered question never left it and the first
wait ran to its timeout; with the created request's type fixed, the first
example's `('pending', 'approval')` went red; with the request itself read
inside the poll loop, the recorded routes held both.
