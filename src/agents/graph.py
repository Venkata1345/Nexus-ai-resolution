from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.agent import generate_response_node
from src.agents.escalation import escalation_node, route_after_router
from src.agents.retriever import retrieve_knowledge_node
from src.agents.router import predict_intent_node
from src.agents.state import NexusState
from src.agents.workers import billing_node, shipping_node


def supervisor_node(state: NexusState):
    """Route to the correct worker based on the predicted intent."""
    print("\n[Supervisor] Analyzing intent to delegate...")
    intent = state.get("intent")

    if intent in ["track_order"]:
        assignee = "shipping"
    elif intent in ["get_refund", "cancel_order", "check_invoices", "track_refund"]:
        assignee = "billing"
    else:
        assignee = "generator"

    print(f"[Supervisor] Delegating to: {assignee}")
    return {"current_assignee": assignee}


def route_to_worker(state: NexusState):
    """Conditional edge after supervisor."""
    return (
        f"{state.get('current_assignee')}_worker"
        if state.get("current_assignee") != "generator"
        else "retriever"
    )


# 1. Initialize graph
workflow = StateGraph(NexusState)

# 2. Nodes
workflow.add_node("router", predict_intent_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("shipping_worker", shipping_node)
workflow.add_node("billing_worker", billing_node)
workflow.add_node("retriever", retrieve_knowledge_node)
workflow.add_node("generator", generate_response_node)

# 3. Flow
workflow.set_entry_point("router")

# After the classifier, either escalate (low confidence) or dispatch.
workflow.add_conditional_edges(
    "router",
    route_after_router,
    {"escalation": "escalation", "supervisor": "supervisor"},
)
workflow.add_edge("escalation", END)

# Supervisor picks a worker (or jumps straight to retriever if no worker needed).
workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "shipping_worker": "shipping_worker",
        "billing_worker": "billing_worker",
        "retriever": "retriever",
    },
)

# Every path converges at the retriever -> generator, so the LLM always
# sees both worker output (if any) and KB context before drafting.
workflow.add_edge("shipping_worker", "retriever")
workflow.add_edge("billing_worker", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

# 4. Checkpointer
memory = MemorySaver()

# 5. Compile with the billing breakpoint preserved.
nexus_app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["billing_worker"],
)
