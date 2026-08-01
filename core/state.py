from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    # LangGraph's reducer safely appends new messages to the existing chat history
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Stores the full extracted English text to be used by the summary node
    transcript: str