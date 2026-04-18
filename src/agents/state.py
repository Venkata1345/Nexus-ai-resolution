from typing import TypedDict, Optional , Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
class NexusState(TypedDict):
    
    messages: Annotated[Sequence[BaseMessage],operator.add] 
    #The raw text of the customer's support ticket
    user_inpput: str
    # The classification predicted by our Phase 1 XGBoost model
    intent: Optional[str]

    current_assignee: Optional[str]

    manager_approved: bool