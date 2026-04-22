"""Nexus chat UI.

A thin frontend over the FastAPI backend. Each conversation lives in its own
thread id; the sidebar surfaces live classifier telemetry, the node trace of
the current turn, production model metadata, and an approve button that
appears whenever the backend has paused at the billing breakpoint.

Run with:  streamlit run app/streamlit_app.py
Assumes the API is up at settings.api_host:settings.api_port.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

from src.config import settings

API_BASE = f"http://{settings.api_host}:{settings.api_port}"

st.set_page_config(page_title="Nexus Support", page_icon=":speech_balloon:", layout="wide")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "threads" not in st.session_state:
    # { thread_id: {"label": str, "messages": [ {role, content} ], "last_response": dict|None} }
    st.session_state.threads = {}
if "active_thread" not in st.session_state:
    st.session_state.active_thread = None


def _new_thread(label: str | None = None) -> str:
    tid = f"ticket-{uuid.uuid4().hex[:8]}"
    st.session_state.threads[tid] = {
        "label": label or f"New ticket ({tid[-4:]})",
        "messages": [],
        "last_response": None,
    }
    st.session_state.active_thread = tid
    return tid


def _on_new_thread_click() -> None:
    """Button callback: creates a thread and updates the selectbox widget.

    Runs BEFORE the next rerun, which is the only time we're allowed to
    modify `thread_selector` (Streamlit freezes widget state after the
    widget has been instantiated in a given run).
    """
    new_tid = _new_thread()
    st.session_state.thread_selector = new_tid


if not st.session_state.threads:
    _new_thread()


# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------


def _post(path: str, payload: dict) -> dict[str, Any]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return {}
    except httpx.HTTPError as e:
        st.error(f"API unreachable: {e}")
        return {}


def _get(path: str) -> dict[str, Any]:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return {}


@st.cache_data(ttl=60)
def get_model_info() -> dict:
    return _get("/model/info")


def _stream_sse(path: str, payload: dict, sink: dict) -> Iterator[str]:
    """Iterator suitable for `st.write_stream`: yields LLM text as it arrives.

    Non-token events (`node`, `done`) are captured into `sink` as side effects
    so the caller can finalize UI state after the stream completes.
    """
    sink["trace"] = []
    sink["final"] = None
    try:
        with httpx.stream(
            "POST", f"{API_BASE}{path}", json=payload, timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                kind = data.get("event")
                if kind == "token":
                    yield data.get("text", "")
                elif kind == "node":
                    sink["trace"].append(data)
                elif kind == "done":
                    sink["final"] = data
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        st.error(f"API unreachable: {e}")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("Nexus")

    # Thread selector
    st.subheader("Conversations")
    thread_labels = {tid: t["label"] for tid, t in st.session_state.threads.items()}
    chosen = st.selectbox(
        "Active thread",
        options=list(thread_labels.keys()),
        format_func=lambda tid: thread_labels.get(tid, tid),
        index=list(thread_labels.keys()).index(st.session_state.active_thread),
        key="thread_selector",
    )
    st.session_state.active_thread = chosen
    st.button(
        "+ New conversation",
        use_container_width=True,
        on_click=_on_new_thread_click,
    )

    st.divider()

    # Current-turn telemetry
    st.subheader("Last turn")
    active = st.session_state.threads[st.session_state.active_thread]
    resp = active["last_response"]
    if resp:
        c_intent = resp.get("intent") or "-"
        c_conf = resp.get("intent_confidence")
        c_assignee = resp.get("current_assignee") or "-"
        c_status = resp.get("status") or "-"

        st.markdown(f"**Status:** `{c_status}`")
        st.markdown(f"**Intent:** `{c_intent}`")
        if c_conf is not None:
            st.progress(min(max(c_conf, 0.0), 1.0), text=f"confidence {c_conf:.3f}")
        st.markdown(f"**Assignee:** `{c_assignee}`")

        with st.expander("Node trace"):
            for step in resp.get("trace", []) or []:
                name = step.get("node", "?")
                preview = (step.get("message_preview") or "").strip()
                line = f"- **{name}**"
                if step.get("intent"):
                    line += (
                        f" _(intent={step['intent']}, conf={step.get('intent_confidence', 0):.2f})_"
                    )
                elif step.get("current_assignee"):
                    line += f" _(assignee={step['current_assignee']})_"
                st.markdown(line)
                if preview:
                    st.caption(preview[:180] + ("..." if len(preview) > 180 else ""))
    else:
        st.caption("No turns yet. Send a message to see telemetry.")

    st.divider()

    # Production model metadata
    st.subheader("Production model")
    info = get_model_info()
    if info:
        st.markdown(f"**Name:** `{info.get('registered_name')}`")
        st.markdown(f"**Version:** `{info.get('version')}` @ `{info.get('alias')}`")
        metrics = info.get("metrics") or {}
        if "val_f1_weighted" in metrics:
            st.metric("val F1", f"{metrics['val_f1_weighted']:.4f}")
        if "test_f1_weighted" in metrics:
            st.metric("test F1", f"{metrics['test_f1_weighted']:.4f}")
    else:
        st.caption("Model registry unreachable.")


# -----------------------------------------------------------------------------
# Main chat panel
# -----------------------------------------------------------------------------

st.title("Nexus Support")
st.caption("Agentic customer-support demo: XGBoost routing + RAG + human approval gate.")

active = st.session_state.threads[st.session_state.active_thread]

for msg in active["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _finalize_turn(full_text: str, sink: dict) -> None:
    """Update session state after a stream completes."""
    final = sink.get("final") or {}
    trace = sink.get("trace", [])
    active["last_response"] = {
        "status": final.get("status", "complete"),
        "intent": final.get("intent"),
        "intent_confidence": final.get("intent_confidence"),
        "current_assignee": final.get("current_assignee"),
        "paused_before": final.get("paused_before"),
        "trace": trace,
    }
    # Prefer the streamed text; fall back to any `reply` on the final frame
    # (approve-when-already-complete path emits no tokens, just the reply).
    text = full_text or final.get("reply") or ""
    if final.get("status") != "awaiting_approval" and text:
        active["messages"].append({"role": "assistant", "content": text})


# If the backend paused at the billing worker, surface an approve button.
resp = active["last_response"] or {}
if resp.get("status") == "awaiting_approval":
    st.warning(
        f"Financial action pending ({resp.get('intent')}). "
        "A human manager must approve before proceeding."
    )
    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("Approve", type="primary"):
            with st.chat_message("assistant"):
                sink: dict = {}
                full_text = st.write_stream(
                    _stream_sse(
                        f"/chat/{st.session_state.active_thread}/approve/stream",
                        {"approved": True},
                        sink,
                    )
                )
            _finalize_turn(full_text or "", sink)
            st.rerun()
    with cols[1]:
        if st.button("Reject"):
            # No backend reject flow yet; just post a canned message client-side.
            active["messages"].append(
                {
                    "role": "assistant",
                    "content": "Request rejected by manager. A human agent will follow up.",
                }
            )
            active["last_response"]["status"] = "complete"
            st.rerun()

if prompt := st.chat_input("Type your support request..."):
    active["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        sink = {}
        full_text = st.write_stream(
            _stream_sse(
                "/chat/stream",
                {"thread_id": st.session_state.active_thread, "message": prompt},
                sink,
            )
        )
    _finalize_turn(full_text or "", sink)
    st.rerun()
