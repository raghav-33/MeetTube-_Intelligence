from core.state import GraphState
from utils.llm import get_llm
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq


# 1. Structured Output Schema
class RouteDecision(BaseModel):
    destination: str = Field(
        description="The target node. Must be exactly 'rag_executor', 'summary_executor', or 'direct_chat_executor'."
    )

def semantic_router(state: GraphState) -> str:
    """ Intelligently routes the user's query Using Output Recievef From LLM"""
    print("Evaluating query intent...")
    
    
    
    # 2. Force the LLM to output our Pydantic schema
    base_llm = router_llm = ChatMistralAI(
        model="open-mistral-7b",  # Blazing fast compared to mistral-large
        temperature=0,            # Force deterministic choices
        max_tokens=15             # Prevents the model from rambling, saving massive completion time
    )
    llm = base_llm.with_structured_output(RouteDecision)
    
    # prompt 
    system_prompt = (
        "You are the Intelligent routing engine for an AI assistant. Analyze the user's latest message. and route it to one of following exact keys \n"
        "'summary_executor': If the user explicitly asks for a summary, key insights, action items, "
        "or an overview of the entire document/transcript.\n\n"
        
        "'rag_executor': If the user is asking a specific factual question about the contents, data, "
        "decisions, or facts contained inside the uploaded document/transcript.\n\n"
        
        "'direct_chat_executor': If the user is greeting you, making small talk, or asking meta-questions "
        "about the conversation itself (e.g., 'What did I just say?', 'Clear chat history', 'Who are you?').\n\n"
        
        "Output ONLY the raw string key ('summarize', 'rag', or 'direct'). Do not include explanation."
            
    )
    
    # extracting User's latest message
    user_text = state["messages"][-1].content
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text)
    ]
    
    
    
    # 4. Invoke the LLM and extract the string destination
    decision = llm.invoke(messages)
    print(f"Routing to -> {decision.destination.upper()} Node")
    
    return decision.destination