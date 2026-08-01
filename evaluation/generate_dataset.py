import os
import sys
import types

# ─── RAGAS BUG WORKAROUND ───
# Ragas crashes on startup looking for an old Google VertexAI module. 
# We create a fake module here to bypass the error since we use Groq.
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
# ────────────────────────────

from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv
load_dotenv()

def create_ragas_dataset():
    # 1. Load your video transcript 
    # Make sure transcript.txt is saved in the same directory where you run the script!
    loader = TextLoader("transcript.txt")
    documents = loader.load()

    # 2. Set up the LLM and Embeddings for the RAGAS "Teacher"
    raw_llm = ChatGroq(model="llama-3.3-70b-versatile", 
                       temperature=0,
                       model_kwargs={"response_format": {"type": "json_object"}})
    
    raw_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    generator_llm = LangchainLLMWrapper(raw_llm)
    generator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    # 3. Initialize the Testset Generator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    # 4. Generate the Dataset!
    print("Generating test questions and ground truth answers...")
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=25  # Reduced to 5 so it finishes faster while you test
    )

    # 5. Save it to a Pandas DataFrame or CSV for offline evaluation
    df = testset.to_pandas()
    df.to_csv("evaluation/golden_dataset.csv", index=False)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    create_ragas_dataset()