from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.agent import generate_response_node
from src.agents.router import predict_intent_node
from src.agents.state import NexusState
from src.agents.workers import billing_node, shipping_node


def supervisor_node(state: NexusState):
    """
    NODE: The lightweight manager. Reads the XGBoost intent and assigns a worker.
    """
    print("\n[Supervisor] Analyzing intent to delegate...")
    intent = state.get("intent")

    if intent in ["track_order"]:
        assignee = "shipping"
    # --- FIXED: Added 'track_refund' to the billing routing ---
    elif intent in ["get_refund", "cancel_order", "check_invoices", "track_refund"]:
        assignee = "billing"
    else:
        assignee = "generator"

    print(f"[Supervisor] Delegating to: {assignee}")
    return {"current_assignee": assignee}


def route_to_worker(state: NexusState):
    """CONDITIONAL EDGE: Reads the assignee and routes the graph."""
    return (
        f"{state.get('current_assignee')}_worker"
        if state.get("current_assignee") != "generator"
        else "generator"
    )


# 1. Initialize Graph
workflow = StateGraph(NexusState)

# 2. Add Nodes
workflow.add_node("router", predict_intent_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("shipping_worker", shipping_node)
workflow.add_node("billing_worker", billing_node)
workflow.add_node("generator", generate_response_node)

# 3. Define the Flow
workflow.set_entry_point("router")
workflow.add_edge("router", "supervisor")

# The Supervisor conditionally routes to the correct worker
workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {
        "shipping_worker": "shipping_worker",
        "billing_worker": "billing_worker",
        "generator": "generator",
    },
)

# After workers finish looking up data, they always go to the generator to draft the email
workflow.add_edge("shipping_worker", "generator")
workflow.add_edge("billing_worker", "generator")
workflow.add_edge("generator", END)

# 4. Setup Persistence (The Checkpointer)
memory = MemorySaver()

# 5. Compile the Application with a Breakpoint
nexus_app = workflow.compile(
    checkpointer=memory,
    # SECURITY FEATURE: The graph will completely halt right before executing this node
    interrupt_before=["billing_worker"],
)
