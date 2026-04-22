"""Escalation node and gate: low-confidence intents go to a human."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.state import NexusState
from src.config import settings


def route_after_router(state: NexusState) -> str:
    """Conditional edge: escalate if classifier confidence is below threshold."""
    conf = state.get("intent_confidence") or 0.0
    if conf < settings.intent_confidence_threshold:
        print(
            f"[Gate] Confidence {conf:.3f} < {settings.intent_confidence_threshold} "
            f"-- escalating to human."
        )
        return "escalation"
    return "supervisor"


def escalation_node(state: NexusState):
    """Produce a safe fallback response asking the customer to wait for a human."""
    intent = state.get("intent") or "unknown"
    conf = state.get("intent_confidence") or 0.0

    print("\n[Escalation] Producing human-handoff response.")
    msg = AIMessage(
        content=(
            "Thanks for reaching out. I want to make sure you get the right answer, "
            "so I'm routing your message to a human support agent who'll follow up "
            "shortly. "
            f"(Routing diagnostic: predicted intent='{intent}', confidence={conf:.3f}.)"
        )
    )
    return {"messages": [msg]}
