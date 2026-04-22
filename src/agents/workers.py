from langchain_core.messages import SystemMessage

from src.agents.state import NexusState

# Domain-Specific Mock Databases
SHIPPING_DB = {
    "track_order": "Order #999-TX is out for delivery. Arriving today by 8 PM.",
}

BILLING_DB = {
    "get_refund": "Refund of $42.50 has been successfully routed back to the original Visa card.",
    "track_refund": "Refund of $42.50 is currently processing and will appear in 3-5 business days.",
    "check_invoices": "Your last invoice was for $42.50 on March 1st. Status: Paid.",
}


def shipping_node(state: NexusState):
    """
    WORKER: The Shipping Agent.
    Handles tracking and logistics. Safe to run autonomously.
    """
    print("\n[Shipping Worker] Accessing secure logistics database...")
    intent = state.get("intent")

    # Fetch data or use a fallback
    info = SHIPPING_DB.get(intent, "No shipping records found for this specific query.")

    # We append this information to the LangGraph memory as a "System Message"
    # so the final LLM knows exactly what the database said.
    system_note = SystemMessage(content=f"Shipping Database Result: {info}")

    print(f"[Shipping Worker] Retrieved data: {info}")
    return {"messages": [system_note]}


def billing_node(state: NexusState):
    """
    WORKER: The Billing Agent.
    Handles secure financial requests. Strictly requires Human-in-the-Loop for refunds.
    """
    print("\n[Billing Worker] Intercepted financial request...")
    intent = state.get("intent")

    # SECURITY GATE: Check if a human manager has approved this transaction
    if intent in ["get_refund", "cancel_order"]:
        if not state.get("manager_approved", False):
            # If the human hasn't approved it yet, we block the action.
            print("[Billing Worker] HALT: Manager approval required for financial actions.")
            warning = SystemMessage(
                content="Billing Tool Result: TRANSACTION BLOCKED. Awaiting human manager approval to process refund."
            )
            return {"messages": [warning]}

        else:
            # If the human flipped the flag, we process the money!
            print("[Billing Worker] SUCCESS: Human approval detected. Processing funds.")
            info = BILLING_DB.get(intent, "Transaction complete.")
            success = SystemMessage(content=f"Billing Tool Result: {info}")
            return {"messages": [success]}

    # If it's just a harmless question (like checking an invoice), no human needed
    safe_info = BILLING_DB.get(intent, "Billing records clear.")
    return {"messages": [SystemMessage(content=f"Billing Tool Result: {safe_info}")]}
