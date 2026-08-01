from core.state import GraphState
from utils.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def summary_executor(state: GraphState) -> dict:
    """Create a summary and key points for meetings or YouTube videos."""
    print("Executing Adaptive Summary & Insights Node...")
    
    # Get Transcript 
    transcript = state.get("transcript", "")
    
    if not transcript:
        return {
            "messages": [
                AIMessage(content="I couldn't find an active transcript. Please process a video or audio file first!")
            ]
        }
    
    system_prompt = (
        "You are an expert content and meeting analyst. Read the provided transcript and generate a structured report.\n\n"
        "Part 1: Executive Overview\n"
        "Provide a concise summary of the primary topic, core themes, and the overall purpose of the audio.\n\n"
        "Part 2: Action Items or Key Takeaways\n"
        "First, determine the type of content. Then follow the matching formatting rule below:\n\n"
        "CRITERIA A: If the transcript is a formal meeting or team discussion:\n"
        "Extract all action items as a numbered list. For each item, explicitly provide:\n"
        "  - Task description\n"
        "  - Owner (who is responsible)\n"
        "  - Deadline (if mentioned, else write 'Not specified')\n\n"
        "CRITERIA B: If the transcript is a YouTube video, tutorial, lecture, or presentation:\n"
        "Extract the core concepts, main arguments, and actionable takeaways as a structured bulleted list. "
        "For each takeaway, provide a 1-sentence explanation of its significance.\n\n"
        "Do not include any introductory conversational fluff.\n\n"
        "Source Transcript:\n{transcript_data}"
    )
    
    # Extract the user's latest input
    user_text = state["messages"][-1].content
    
    # Assemble the payload with strict message typing
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])

    # LLM 
    llm = get_llm()
    
    # Output Parser 
    output_parser = StrOutputParser()
    
    # Chain
    chain = prompt | llm | output_parser
    
    # Chain Invoke
    response_text = chain.invoke({
        "transcript_data": transcript,
        "user_input": user_text
    })
    
    return {"messages": [AIMessage(content=response_text)]}

