from src.agents.state import NexusState

# Our mock enterprise database
# In a real system, this would be SQL queries or API calls to Shopify/Salesforce
MOCK_DATABASE = {
    "track_order": "Order #999-TX is currently out for delivery and will arrive by 8 PM.",
    "cancel_order": "Order #999-TX is eligible for cancellation. Please click the link in your email to confirm.",
    "get_refund": "Account Status: Eligible for refund. No active refund requests are currently pending.",
    "check_invoices": "Your last invoice was for $42.50 on March 1st. Status: Paid in full.",
}


def execute_action_node(state: NexusState):
    """
    NODE: Acts as a simulated database lookup.
    It reads the predicted intent from the state and fetches the relevant customer data.
    """
    intent = state.get("intent")
    print(f"\n[Action Node] Accessing secure database for intent: '{intent}'...")

    # Fetch the data from our mock database, or return a fallback message
    fetched_context = MOCK_DATABASE.get(
        intent, "System Note: No specific account records found for this request."
    )

    print(f"[Action Node] Retrieved context: {fetched_context}")

    # Return the updated state payload.
    # LangGraph will automatically inject this into the shared memory.
    return {"context": fetched_context}
