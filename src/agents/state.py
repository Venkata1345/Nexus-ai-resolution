import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class NexusState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The raw text of the customer's support ticket
    user_input: str
    # The classification predicted by our Phase 1 XGBoost model
    intent: str | None

    current_assignee: str | None

    manager_approved: bool
