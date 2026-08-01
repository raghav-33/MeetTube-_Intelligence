from langgraph.graph import StateGraph, END
from core.state import GraphState
from agents.router import semantic_router
from agents.direct_node import direct_chat_executor
from agents.rag_node import rag_executor
from agents.summarizer import summary_executor
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


def build_graph():
    """
    Constructs the LangGraph state machine, maps the routing logic, 
    and attaches a local SQLite database for persistent conversational memory.
    """
    print("Initializing LangGraph Orchestrator...")
    
    # 1. Initialize the Graph with your custom State schema
    workflow = StateGraph(GraphState)
    
    # 2. Add the three execution nodes
    workflow.add_node("direct_chat_executor", direct_chat_executor)
    workflow.add_node("rag_executor", rag_executor)
    workflow.add_node("summary_executor", summary_executor)
    
    # 3. Add the Conditional Entry Point (The Router)
    # The router looks at the user's message and returns a string.
    # The dictionary maps that string to the exact node to execute.
    workflow.set_conditional_entry_point(
        semantic_router,
        {
            "direct_chat_executor": "direct_chat_executor",
            "rag_executor": "rag_executor",
            "summary_executor": "summary_executor"
        }
    )
    
    # 4. Route all nodes to END 
    workflow.add_edge("direct_chat_executor", END)
    workflow.add_edge("rag_executor", END)
    workflow.add_edge("summary_executor", END)
    
    # 5. Set up SQLite Persistence
    # check_same_thread=False is strictly required when using Streamlit 
    # because Streamlit heavily utilizes multi-threading under the hood.
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # 6. Compile the graph with memory attached
    app = workflow.compile(checkpointer=memory)
    print(" Graph successfully compiled with SQLite persistence.")
    
    return app

# Initialize a global instance of the app so Streamlit can import it easily
graph_app = build_graph()