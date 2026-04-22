"""Tests for the confidence gate and escalation node."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.escalation import escalation_node, route_after_router
from src.config import settings


def test_route_above_threshold_goes_to_supervisor():
    state = {"intent_confidence": settings.intent_confidence_threshold + 0.1}
    assert route_after_router(state) == "supervisor"


def test_route_at_threshold_goes_to_supervisor():
    state = {"intent_confidence": settings.intent_confidence_threshold}
    # Threshold is inclusive: exactly at threshold counts as confident enough.
    assert route_after_router(state) == "supervisor"


def test_route_below_threshold_goes_to_escalation():
    state = {"intent_confidence": settings.intent_confidence_threshold - 0.01}
    assert route_after_router(state) == "escalation"


def test_route_missing_confidence_treated_as_zero():
    state: dict = {}
    # No confidence recorded at all -> safer to escalate than to dispatch blind.
    assert route_after_router(state) == "escalation"


def test_route_zero_confidence_escalates():
    assert route_after_router({"intent_confidence": 0.0}) == "escalation"


def test_escalation_node_returns_ai_message():
    state = {"intent": "track_order", "intent_confidence": 0.12}
    result = escalation_node(state)
    assert "messages" in result
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)


def test_escalation_message_mentions_human_and_diagnostics():
    state = {"intent": "get_refund", "intent_confidence": 0.25}
    msg = escalation_node(state)["messages"][0]
    text = msg.content.lower()
    assert "human" in text
    assert "get_refund" in msg.content
    assert "0.250" in msg.content  # the formatted confidence


def test_escalation_handles_missing_state_fields():
    # Even if upstream passes nothing, the node shouldn't crash.
    msg = escalation_node({})["messages"][0]
    assert "unknown" in msg.content
    assert "0.000" in msg.content
