from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.state import NexusState
from src.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    temperature=settings.llm_temperature,
    google_api_key=settings.gemini_api_key.get_secret_value(),
)


def generate_response_node(state: NexusState):
    """Draft the final customer-facing reply by feeding the full thread to the LLM."""
    print("\n[Generator] Drafting final response...")

    system_instruction = SystemMessage(
        content=(
            "You are Nexus, a professional customer-support AI. "
            "Read the conversation thread carefully. You may see two kinds of "
            "context messages:\n"
            "  1. 'Database Result' lines from internal workers (shipping/billing). "
            "Treat these as authoritative facts -- use them verbatim where relevant.\n"
            "  2. 'KNOWLEDGE BASE RESULTS' with retrieved past Q&A pairs. Use these "
            "as examples of how we usually respond; paraphrase rather than copy, and "
            "only rely on them when no Database Result is available.\n"
            "Never invent tracking numbers, order IDs, or policies that aren't in the "
            "provided context. Be concise and polite."
        )
    )

    conversation_thread = [system_instruction, *state["messages"]]
    response = llm.invoke(conversation_thread)

    print("[Generator] Response drafted successfully.")
    return {"messages": [response]}
