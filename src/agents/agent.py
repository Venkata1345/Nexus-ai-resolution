from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.state import NexusState

llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-3.1-flash-lite-preview")


def generate_response_node(state: NexusState):
    """
    NODE: The final LLM generator.
    Reads the entire message thread (including worker notes) and replies to the user.
    """
    print("\n[Generator] Drafting final response...")

    # Give the LLM its core identity
    system_instruction = SystemMessage(
        content="You are Nexus, a highly professional customer support AI. "
        "Read the conversation thread below. If a worker agent provided a Database Result, "
        "use that exact information to answer the user. Do not make up tracking numbers or policies."
    )

    # Combine our system instruction with the running chat history
    conversation_thread = [system_instruction] + state["messages"]

    # Generate the response
    response = llm.invoke(conversation_thread)

    print("[Generator] Response drafted successfully.")

    # Append the final AI response to the state
    return {"messages": [response]}
