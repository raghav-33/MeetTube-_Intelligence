import os
import ast
import types
import sys
import pandas as pd
from dotenv import load_dotenv

# 1. RAGAS IMPORT CRASH BYPASS
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy 
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

load_dotenv()

def run_retrieval_evaluation():
    # 2. Setup Models
    # FIX A: Switched to llama-3.1-8b-instant (5x higher daily token limit)
    # FIX B: Added model_kwargs={"n": 1} to stop the 'BadRequestError' on answer_relevancy
    base_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        n=1
    )
    llm = LangchainLLMWrapper(base_llm)
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))

    # FIX C: Relax answer_relevancy strictness to avoid requesting n > 1 from non-OpenAI endpoints
    answer_relevancy.strict = False
    answer_relevancy.strictness = 1

    # 3. Load Dataset
    df = pd.read_csv("evaluation/dataset_ready_for_eval.csv")

    # 4. Map directly to Ragas format
    samples = []
    for _, row in df.iterrows():
        # Safely parse stringified list or use raw list
        if isinstance(row['reference_contexts'], str):
            contexts = ast.literal_eval(row['reference_contexts'])
        else:
            contexts = row['reference_contexts']
            
        # Ensure answer is stringified safely
        ai_response = str(row['answer']) if pd.notna(row['answer']) else ""

        sample = SingleTurnSample(
            user_input=row['user_input'],
            retrieved_contexts=contexts,
            reference=row['reference'],
            response=ai_response
        )
        samples.append(sample)

    # 5. Evaluate
    print("Evaluating Retrieval & Generation Metrics...")
    results = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1) # Prevents Groq rate limits
    )
    
    print("\n--- Results ---")
    print(results)

if __name__ == "__main__":
    run_retrieval_evaluation()