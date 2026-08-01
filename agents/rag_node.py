from core.state import GraphState
from utils.llm import get_llm
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from RAG.rag import AdvancedHybridRAG

# Initialize the retriever globally so it persists in memory while the app runs
retriever = AdvancedHybridRAG()

def format_context_blocks(retrieved_docs) -> str:
    """Helper to convert list of retrieved docs/dicts into clean text."""
    if not retrieved_docs:
        return ""
    
    cleaned_chunks = []
    for item in retrieved_docs:
        # Check if item is a LangChain Document or a Dict/String
        if hasattr(item, 'page_content'):
            cleaned_chunks.append(item.page_content)
        elif isinstance(item, dict) and 'text' in item:
            cleaned_chunks.append(item['text'])
        else:
            cleaned_chunks.append(str(item))
            
    return "\n\n---\n\n".join(cleaned_chunks)

def rag_executor(state: GraphState) -> dict:
    """
    Executes the Hybrid RAG pipeline using LCEL. Retrieves the best chunks 
    and generates a grounded answer based exclusively on the context.
    """
    print("Executing Hybrid RAG Node...")
    
    # 1. Extract the user's latest question
    user_query = state["messages"][-1].content
    
    # 2. Retrieve the top 4 highly relevant chunks using FlashRank + BM25 + Chroma
    raw_context = retriever.retrieve_and_rerank(user_query, final_k=4)
    
    
    if ("No content ingested yet" in str(raw_context) or not raw_context) and state.get("transcript"):
        print("🔄 Syncing graph database instance with current video transcript...")
        retriever.ingest_transcript(state["transcript"])
        # Re-run the query now that data is loaded
        raw_context = retriever.retrieve_and_rerank(user_query, final_k=4)
    
    # 3. Format raw chunks into a single clean string
    context_str = format_context_blocks(raw_context)

    # Debug log to verify context isn't empty during evaluation
    if not context_str.strip():
        print("⚠️ WARNING: Context is empty! Ensure documents/transcripts are ingested into ChromaDB before evaluating.")
    
    # 3. Define the strict, anti-hallucination prompt template
    system_prompt = (
        """
        You are a precise and expert video data analyst. Answer the user's question using ONLY the context provided below.\n
        The context contains timestamp tags in the format [MM:SS - MM:SS]. 
        When you provide facts, quotes, or summaries, you MUST cite the exact timestamp from the context so the user knows where to look in the video. 
        Example response: "The speaker discusses the new AI architecture starting at [04:00 - 06:00]."
        If the answer is not contained in the context, explicitly state: 
        I cannot find the answer to this in the transcript.' Do not guess or use outside knowledge.\n\n
        Context Blocks:\n{context_data}
        """
    )
    
    # 4. Build the prompt blueprint
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # 5. Define the String Output Parser
    output_parser = StrOutputParser()
    
    # 6. Build the LCEL Chain (Prompt -> LLM -> Parser)
    llm = get_llm()
    chain = prompt | llm | output_parser
    
    # 7. Invoke the chain by passing the dynamic variables
    response_text = chain.invoke({
        "context_data": context_str,
        "question": user_query
    })
    
    # 8. Wrap the raw string response back into an AIMessage for LangGraph's memory
    return {"messages": [AIMessage(content=response_text)]}