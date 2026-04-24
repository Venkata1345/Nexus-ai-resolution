"""Nexus FastAPI backend.

Exposes the agentic graph as a stateful HTTP service with per-thread
checkpointing. The graph itself is imported from src.agents.graph and is
stepped via FastAPI endpoints that mirror what main.py does interactively:
  - POST /chat              : run until completion OR the approval breakpoint
  - POST /chat/{id}/approve : resume a paused graph after manager approval
  - GET  /chat/{id}         : fetch the visible message history

Classifier telemetry (intent, confidence, assignee, node trace) is attached
to every response so the Streamlit UI can render a live sidebar without
making extra calls.
"""

from __future__ import annotations

from dotenv import load_dotenv

# Must precede any import that instantiates the Gemini client.
load_dotenv()

import json  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402
from starlette.responses import JSONResponse, StreamingResponse  # noqa: E402

from src.agents.graph import nexus_app  # noqa: E402
from src.api.deps import limiter, logger  # noqa: E402
from src.api.schemas import (  # noqa: E402
    ApproveRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    MessageOut,
    ModelInfoResponse,
    NodeEvent,
)
from src.config import settings  # noqa: E402

# -----------------------------------------------------------------------------
# App + middleware
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api.startup", thread_count=0)
    yield
    logger.info("api.shutdown")


app = FastAPI(
    title="Nexus Intent Agent API",
    description=(
        "Stateful agentic customer-support backend. Routes with XGBoost + MiniLM, "
        "grounds replies with RAG, and enforces human approval on financial actions."
    ),
    version="0.4.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_text(content) -> str:
    """Normalize a BaseMessage.content value into a plain string.

    Newer Gemini responses come back as a list of content parts; older
    LangChain messages are plain strings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(chunks)
    return str(content)


def _visible_messages(messages) -> list[MessageOut]:
    """Project graph messages to the user-visible role/content shape."""
    out: list[MessageOut] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(MessageOut(role="user", content=_extract_text(m.content)))
        elif isinstance(m, AIMessage):
            out.append(MessageOut(role="assistant", content=_extract_text(m.content)))
        elif isinstance(m, SystemMessage):
            # Internal machinery (worker DB results, RAG context) — hide from
            # the UI history but kept in graph state for the LLM.
            continue
    return out


def _snapshot_telemetry(snapshot) -> dict:
    values = snapshot.values or {}
    return {
        "intent": values.get("intent"),
        "intent_confidence": values.get("intent_confidence"),
        "current_assignee": values.get("current_assignee"),
    }


def _collect_trace(events: list[dict]) -> list[NodeEvent]:
    """Turn a list of stream_mode='updates' events into a user-facing trace."""
    trace: list[NodeEvent] = []
    for ev in events:
        # Each event is a dict keyed by node name -> that node's state delta.
        for node, delta in ev.items():
            preview = None
            if isinstance(delta, dict):
                msgs = delta.get("messages") or []
                if msgs:
                    preview = _extract_text(msgs[-1].content)[:200]
            trace.append(
                NodeEvent(
                    node=node,
                    intent=(delta or {}).get("intent"),
                    intent_confidence=(delta or {}).get("intent_confidence"),
                    current_assignee=(delta or {}).get("current_assignee"),
                    message_preview=preview,
                )
            )
    return trace


def _derive_status(snapshot) -> tuple[str, str | None]:
    """Classify a snapshot as complete / awaiting_approval / escalated."""
    next_nodes = list(snapshot.next or [])
    if "billing_worker" in next_nodes:
        return "awaiting_approval", "billing_worker"
    # If the last message is from the escalation node the graph is done but
    # the response is a handoff, which the UI may want to display differently.
    values = snapshot.values or {}
    msgs = values.get("messages") or []
    if (
        msgs
        and isinstance(msgs[-1], AIMessage)
        and "human support agent" in _extract_text(msgs[-1].content).lower()
    ):
        return "escalated", None
    return "complete", None


def _last_ai_reply(snapshot) -> str:
    values = snapshot.values or {}
    for m in reversed(values.get("messages") or []):
        if isinstance(m, AIMessage):
            return _extract_text(m.content)
    return ""


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(model_loaded=True, kb_loaded=True)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model metadata from the bundle if present, else from MLflow.

    Deployed containers ship a `bundle/metadata.json` produced by
    `python -m src.router.bundle`, which lets us answer this endpoint without
    needing the mlflow client (or the SQLite DB) at runtime.
    """
    bundle_meta = settings.bundle_dir / "metadata.json"
    if bundle_meta.exists():
        meta = json.loads(bundle_meta.read_text())
        return ModelInfoResponse(
            registered_name=meta["registered_name"],
            version=str(meta["version"]),
            alias=meta["alias"],
            run_id=meta.get("run_id"),
            metrics=meta.get("metrics", {}),
        )

    # Local-dev fallback: pull live from MLflow.
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        mv = client.get_model_version_by_alias(
            settings.mlflow_registered_model_name, settings.mlflow_production_alias
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No production model: {exc}") from exc

    metrics: dict[str, float] = {}
    if mv.run_id:
        run = client.get_run(mv.run_id)
        metrics = {k: v for k, v in run.data.metrics.items() if isinstance(v, (int, float))}

    return ModelInfoResponse(
        registered_name=mv.name,
        version=str(mv.version),
        alias=settings.mlflow_production_alias,
        run_id=mv.run_id,
        metrics=metrics,
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.api_rate_limit)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    config = _thread_config(body.thread_id)
    logger.info("chat.request", thread_id=body.thread_id, len=len(body.message))

    inputs = {"messages": [HumanMessage(content=body.message)]}
    events: list[dict] = []
    for ev in nexus_app.stream(inputs, config, stream_mode="updates"):
        events.append(ev)

    snapshot = nexus_app.get_state(config)
    status, paused_before = _derive_status(snapshot)
    telemetry = _snapshot_telemetry(snapshot)
    trace = _collect_trace(events)

    logger.info(
        "chat.response",
        thread_id=body.thread_id,
        status=status,
        intent=telemetry["intent"],
        confidence=telemetry["intent_confidence"],
    )

    return ChatResponse(
        thread_id=body.thread_id,
        status=status,
        reply="" if status == "awaiting_approval" else _last_ai_reply(snapshot),
        paused_before=paused_before,
        trace=trace,
        **telemetry,
    )


@app.post("/chat/{thread_id}/approve", response_model=ChatResponse)
@limiter.limit(settings.api_rate_limit)
def approve(request: Request, thread_id: str, body: ApproveRequest) -> ChatResponse:
    """Approve a paused billing action and resume the graph.

    Idempotent: if the thread is no longer paused (e.g. an earlier duplicate
    click already drove it to completion), we return the current state with
    the final reply instead of 409. This makes Streamlit's "click again
    while the LLM is still drafting" behaviour harmless.

    Returns 404 only if we've never seen this thread at all.
    """
    if not body.approved:
        # Future: add an explicit "reject" path that routes to escalation.
        raise HTTPException(status_code=501, detail="Rejection flow not yet implemented.")

    config = _thread_config(thread_id)
    snapshot = nexus_app.get_state(config)

    is_paused = "billing_worker" in (snapshot.next or ())
    has_history = bool((snapshot.values or {}).get("messages"))

    if not is_paused and not has_history:
        raise HTTPException(status_code=404, detail=f"Unknown thread {thread_id}.")

    events: list[dict] = []
    if is_paused:
        nexus_app.update_state(config, {"manager_approved": True})
        for ev in nexus_app.stream(None, config, stream_mode="updates"):
            events.append(ev)
        snapshot = nexus_app.get_state(config)

    status, paused_before = _derive_status(snapshot)
    telemetry = _snapshot_telemetry(snapshot)

    logger.info(
        "chat.approve",
        thread_id=thread_id,
        status=status,
        intent=telemetry["intent"],
        was_paused=is_paused,
    )

    return ChatResponse(
        thread_id=thread_id,
        status=status,
        reply=_last_ai_reply(snapshot) if status != "awaiting_approval" else "",
        paused_before=paused_before,
        trace=_collect_trace(events),
        **telemetry,
    )


@app.get("/chat/{thread_id}", response_model=HistoryResponse)
def history(thread_id: str) -> HistoryResponse:
    snapshot = nexus_app.get_state(_thread_config(thread_id))
    values = snapshot.values or {}
    return HistoryResponse(
        thread_id=thread_id,
        messages=_visible_messages(values.get("messages") or []),
    )


# =============================================================================
# Streaming endpoints (Server-Sent Events)
# =============================================================================


def _sse(event_obj: dict) -> str:
    """Format a dict as a single SSE `data: ...\\n\\n` frame."""
    return f"data: {json.dumps(event_obj)}\n\n"


def _stream_graph(inputs: dict | None, config: dict):
    """Drive the graph and yield SSE frames for node events + LLM tokens.

    Uses LangGraph's multi-mode streaming so we get both node-level state
    updates AND token-by-token message chunks from the generator.
    """
    node_events: list[dict] = []

    for chunk in nexus_app.stream(inputs, config, stream_mode=["updates", "messages"]):
        mode, payload = chunk

        if mode == "updates":
            for node, delta in payload.items():
                if isinstance(delta, dict):
                    preview = None
                    msgs = delta.get("messages") or []
                    if msgs:
                        preview = _extract_text(msgs[-1].content)[:200]
                    event = {
                        "event": "node",
                        "node": node,
                        "intent": delta.get("intent"),
                        "intent_confidence": delta.get("intent_confidence"),
                        "current_assignee": delta.get("current_assignee"),
                        "message_preview": preview,
                    }
                else:
                    event = {"event": "node", "node": node}
                node_events.append(event)
                yield _sse(event)

        elif mode == "messages":
            message_chunk, metadata = payload
            # Only stream tokens from the final generator; worker SystemMessages
            # also flow through this mode but are internal machinery.
            if metadata.get("langgraph_node") != "generator":
                continue
            text = _extract_text(message_chunk.content)
            if text:
                yield _sse({"event": "token", "text": text})

    snapshot = nexus_app.get_state(config)
    status, paused_before = _derive_status(snapshot)
    telemetry = _snapshot_telemetry(snapshot)
    yield _sse(
        {
            "event": "done",
            "status": status,
            "paused_before": paused_before,
            "reply": _last_ai_reply(snapshot) if status != "awaiting_approval" else "",
            **telemetry,
            "trace": node_events,
        }
    )


@app.post("/chat/stream")
@limiter.limit(settings.api_rate_limit)
def chat_stream(request: Request, body: ChatRequest):
    """Streaming variant of /chat — emits SSE events for nodes + LLM tokens."""
    config = _thread_config(body.thread_id)
    logger.info("chat.stream.request", thread_id=body.thread_id, len=len(body.message))
    inputs = {"messages": [HumanMessage(content=body.message)]}
    return StreamingResponse(
        _stream_graph(inputs, config),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},  # hint to any proxy not to buffer
    )


@app.post("/chat/{thread_id}/approve/stream")
@limiter.limit(settings.api_rate_limit)
def approve_stream(request: Request, thread_id: str, body: ApproveRequest):
    """Streaming variant of /approve — same idempotency rules as the JSON endpoint."""
    if not body.approved:
        raise HTTPException(status_code=501, detail="Rejection flow not yet implemented.")

    config = _thread_config(thread_id)
    snapshot = nexus_app.get_state(config)
    is_paused = "billing_worker" in (snapshot.next or ())
    has_history = bool((snapshot.values or {}).get("messages"))

    if not is_paused and not has_history:
        raise HTTPException(status_code=404, detail=f"Unknown thread {thread_id}.")

    if not is_paused:
        # Already complete — emit a single "done" frame and bail.
        def _done_only():
            status, paused_before = _derive_status(snapshot)
            telemetry = _snapshot_telemetry(snapshot)
            yield _sse(
                {
                    "event": "done",
                    "status": status,
                    "paused_before": paused_before,
                    "reply": _last_ai_reply(snapshot),
                    **telemetry,
                    "trace": [],
                }
            )

        return StreamingResponse(_done_only(), media_type="text/event-stream")

    nexus_app.update_state(config, {"manager_approved": True})
    logger.info("chat.approve.stream", thread_id=thread_id)
    return StreamingResponse(
        _stream_graph(None, config),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
