import os
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


def get_llm():
    "Initializes and returns the LLM "
    
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ GROQ_API_KEY is missing from your .env file.")
    
    # Initalize the LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2 
        )
    
    return llm
