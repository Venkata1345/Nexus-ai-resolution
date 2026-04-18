from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_core.messages import HumanMessage
from src.agents.graph import nexus_app

# The checkpointer needs a thread ID so it knows which conversation to remember
config = {"configurable": {"thread_id": "ticket_101"}}

def run_nexus():
    print("=== WELCOME TO NEXUS SUPPORT ===")
    
    # 1. The User asks for a refund
    initial_input = {"messages": [HumanMessage(content="I need a refund for my order please.")]}
    print("\nUSER: I need a refund for my order please.")
    
    # 2. Run the graph until it hits the breakpoint
    print("\n--- INITIATING AI PIPELINE ---")
    for event in nexus_app.stream(initial_input, config, stream_mode="values"):
        pass 
    
    # 3. Check the graph state
    snapshot = nexus_app.get_state(config)
    next_step = snapshot.next
    
    if "billing_worker" in next_step:
        print("\n🛑 GRAPH PAUSED: HUMAN-IN-THE-LOOP REQUIRED")
        print("A financial action has been requested and paused by the Checkpointer.")
        
        # 4. The Human Manager manually approves the transaction by updating the state
        approval = input("\nType 'approve' to authorize this refund: ")
        
        if approval.lower() == 'approve':
            print("\n✅ Manager approval granted. Injecting into state and resuming graph...")
            nexus_app.update_state(config, {"manager_approved": True})
            
            # Resume the graph from where it paused with None
            for event in nexus_app.stream(None, config, stream_mode="values"):
                pass
                
    # 5. Print final message
    final_state = nexus_app.get_state(config)
    final_message = final_state.values["messages"][-1]
    print(f"\nFINAL NEXUS RESPONSE:\n{final_message.content}")

if __name__ == "__main__":
    run_nexus()