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

    >>> import tempfile, threading
    >>> from contextlib import ExitStack
    >>> from pathlib import Path
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
answers a moment from now, while the tool waits:

    >>> answer = threading.Timer(
    ...     0.3, qmcp_mcp.submit_human_response,
    ...     args=("walkthrough-pin", "approve"),
    ...     kwargs={"responded_by": "walkthrough", "server_url": url})
    >>> answer.start()
    >>> result = qmcp_mcp.await_human_response(
    ...     "walkthrough-pin", timeout_seconds=10, poll_seconds=0.05, server_url=url)
    >>> result["status"], result["response"]["response"], result["response"]["responded_by"]
    ('answered', 'approve', 'walkthrough')

`responded_by` is the only trace of how it was answered: the voice loop records
`vox`, a typed answer records whatever the typist gave, and the tool cannot
tell the two apart otherwise.

While the id was still listed as pending the wait read only the listing.
`GET /v1/human/requests/{id}` expires a pending request that is past its
expiry — it is the only thing that does — so a wait that polled it would be
expiring the question it was waiting on. The one read of it happens after the
id has left the listing, and `tests/test_qmcp_mcp.py` holds the wait to that.

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
is an `input`, and the answer is whatever was said:

    >>> def whoever_is_at_the_speaker():
    ...     newest = qmcp_mcp.list_human_requests(status_filter="pending", server_url=url)[0]
    ...     qmcp_mcp.submit_human_response(
    ...         newest["id"], "the pin, not the lock", responded_by="walkthrough", server_url=url)
    >>> threading.Timer(0.3, whoever_is_at_the_speaker).start()
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
example's `('pending', 'approval')` went red.
