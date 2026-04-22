"""Tests for the FastAPI backend. The graph is replaced with a fake so we
can exercise the HTTP contract without touching any ML infrastructure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class _FakeGraph:
    """Stand-in for `nexus_app` with scriptable next-step and values."""

    def __init__(self):
        # What the next get_state() returns. Tests mutate this directly.
        self._next: tuple = ()
        self._messages: list = []
        self._intent: str | None = None
        self._confidence: float | None = None
        self._assignee: str | None = None
        self._manager_approved: bool = False

    # stream(): return a couple of fake update events, then return.
    def stream(self, inputs, config, stream_mode="updates"):
        if inputs is not None:
            # Simulate the router updating intent state + appending a user msg.
            self._messages.append(inputs["messages"][-1])
            yield {
                "router": {
                    "intent": self._intent,
                    "intent_confidence": self._confidence,
                }
            }
            yield {"supervisor": {"current_assignee": self._assignee}}
            if self._assignee == "billing":
                # Pretend we paused at the billing worker.
                self._next = ("billing_worker",)
                return
        # Resuming (inputs is None) or non-billing paths: finish with an AI reply.
        ai = AIMessage(content="final reply")
        self._messages.append(ai)
        self._next = ()
        yield {"generator": {"messages": [ai]}}

    def get_state(self, config):
        return SimpleNamespace(
            next=self._next,
            values={
                "messages": list(self._messages),
                "intent": self._intent,
                "intent_confidence": self._confidence,
                "current_assignee": self._assignee,
                "manager_approved": self._manager_approved,
            },
        )

    def update_state(self, config, updates):
        for k, v in updates.items():
            if k == "manager_approved":
                self._manager_approved = v


@pytest.fixture
def client(monkeypatch):
    import src.api.main as api_main

    fake = _FakeGraph()
    monkeypatch.setattr(api_main, "nexus_app", fake)
    test_client = TestClient(api_main.app)
    test_client.fake = fake  # type: ignore[attr-defined]
    return test_client


# ---------- /health ----------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------- /chat (simple path) ----------


def test_chat_routes_to_generator_and_returns_reply(client):
    client.fake._intent = "check_payment_methods"
    client.fake._confidence = 0.97
    client.fake._assignee = "generator"

    r = client.post("/chat", json={"thread_id": "t1", "message": "how do I pay?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["reply"] == "final reply"
    assert body["intent"] == "check_payment_methods"
    assert body["intent_confidence"] == pytest.approx(0.97)


def test_chat_trace_contains_visited_nodes(client):
    client.fake._intent = "complaint"
    client.fake._confidence = 0.9
    client.fake._assignee = "generator"

    r = client.post("/chat", json={"thread_id": "t-trace", "message": "I am angry"})
    nodes = [step["node"] for step in r.json()["trace"]]
    assert "router" in nodes
    assert "supervisor" in nodes
    assert "generator" in nodes


def test_chat_validates_empty_message(client):
    r = client.post("/chat", json={"thread_id": "t-empty", "message": ""})
    assert r.status_code == 422  # Pydantic min_length=1


# ---------- /chat (billing pause + approve) ----------


def test_chat_pauses_at_billing_and_approval_resumes(client):
    client.fake._intent = "get_refund"
    client.fake._confidence = 0.72
    client.fake._assignee = "billing"

    first = client.post("/chat", json={"thread_id": "t-approve", "message": "refund me"})
    body = first.json()
    assert body["status"] == "awaiting_approval"
    assert body["paused_before"] == "billing_worker"
    assert body["reply"] == ""

    # Approve; graph resumes and completes.
    second = client.post("/chat/t-approve/approve", json={"approved": True})
    body2 = second.json()
    assert body2["status"] == "complete"
    assert body2["reply"] == "final reply"
    assert client.fake._manager_approved is True


def test_approve_404_for_unknown_thread(client):
    # No history, no pause — this thread has never been seen.
    client.fake._next = ()
    client.fake._messages = []
    r = client.post("/chat/nonexistent/approve", json={"approved": True})
    assert r.status_code == 404


def test_approve_is_idempotent_on_completed_thread(client):
    # Thread already completed (e.g. a duplicate click arrived late).
    client.fake._next = ()
    client.fake._messages = [
        HumanMessage(content="refund me"),
        AIMessage(content="final reply"),
    ]
    r = client.post("/chat/t-done/approve", json={"approved": True})
    assert r.status_code == 200
    assert r.json()["status"] == "complete"
    assert r.json()["reply"] == "final reply"


def test_approve_rejection_flow_not_yet_implemented(client):
    client.fake._next = ("billing_worker",)
    r = client.post("/chat/t/approve", json={"approved": False})
    assert r.status_code == 501


# ---------- /chat/{id} history projection ----------


def test_history_hides_system_messages(client):
    client.fake._messages = [
        HumanMessage(content="hi"),
        SystemMessage(content="internal KB context"),
        AIMessage(content="hello!"),
    ]
    r = client.get("/chat/anything")
    msgs = r.json()["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] == "hello!"
