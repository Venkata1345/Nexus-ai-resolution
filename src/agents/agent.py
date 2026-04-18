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
        content="You are Nexus, a highly professional customer support AI. "
        "Read the conversation thread below. If a worker agent provided a Database Result, "
        "use that exact information to answer the user. Do not make up tracking numbers or policies."
    )

    conversation_thread = [system_instruction, *state["messages"]]
    response = llm.invoke(conversation_thread)

    print("[Generator] Response drafted successfully.")
    return {"messages": [response]}
