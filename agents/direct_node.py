
from core.state import GraphState
from utils.llm import get_llm
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

def direct_chat_executor(state: GraphState) -> dict:
    """ Bypasses the RAG pipeline entirely, Handles general knowledge queries, 
        greetings, and casual conversation Which are routed to you.
    """
    print("Executing Direct Chat Node ...")
    
    system_prompt = (
        "You are a highly capable, friendly AI Assistant. Answer the user's general query naturally. "
        "Do not mention video transcripts, meetings, or documents unless the user brings them up."
    )
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history")
    ])
    
    output_parser = StrOutputParser()
    
    llm = get_llm()
    chain = prompt | llm | output_parser
    
    response_text = chain.invoke({
        "chat_history": state["messages"]
    })
    
    return {"messages": [AIMessage(content=response_text)]}
    

'''
but as you are saying for message placeholder user's latest message is already inside the chat_history list but what if user start chat with hello how are you , it will be user's first message and no previous messag exist then what 
'''