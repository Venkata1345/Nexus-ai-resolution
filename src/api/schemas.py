"""Pydantic request/response models for the Nexus API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(
        ..., description="Conversation id; pass a new uuid to start a new thread."
    )
    message: str = Field(..., min_length=1, description="User's latest message.")


class ApproveRequest(BaseModel):
    approved: bool = Field(True, description="Whether the manager approves the paused action.")


class MessageOut(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class NodeEvent(BaseModel):
    node: str
    intent: str | None = None
    intent_confidence: float | None = None
    current_assignee: str | None = None
    message_preview: str | None = None


class ChatResponse(BaseModel):
    """What we return after running (or pausing) the graph for a message."""

    thread_id: str
    status: Literal["complete", "awaiting_approval", "escalated"]
    # Last AI-facing message — what the UI should show the user. Empty string
    # when we're paused before any reply has been drafted.
    reply: str = ""
    # Classifier telemetry for the sidebar.
    intent: str | None = None
    intent_confidence: float | None = None
    current_assignee: str | None = None
    # Which node is paused, if any.
    paused_before: str | None = None
    # Ordered list of node visits during this turn, for the timeline view.
    trace: list[NodeEvent] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageOut]


class ModelInfoResponse(BaseModel):
    registered_name: str
    version: str
    alias: str
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    kb_loaded: bool
