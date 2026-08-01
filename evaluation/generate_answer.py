import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import time


# Add project root to path so Python can find your main app files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CRITICAL: IMPORT YOUR COMPILED LANGGRAPH APP HERE ---
# Change 'my_graph_file' to whatever your python file is named (e.g., app, agent, graph)
# Change 'app' to whatever variable holds your compiled LangGraph
from agents.graph import graph_app 
# ---------------------------------------------------------

load_dotenv()

def generate_pipeline_answers():
    # 1. Load your synthetic golden dataset
    csv_path = "evaluation/golden_dataset.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Make sure the golden dataset exists.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} test queries. Sending to LangGraph...\n")
    
    answers = []
    
    # 2. Loop through every question and send it to your graph
    for idx, row in df.iterrows():
        question = row['user_input']
        print(f"[{idx + 1}/{len(df)}] Asking AI: {question}")
        
        try:
            # Setup thread configuration for persistence (required for most LangGraph setups)
            config = {"configurable": {"thread_id": f"eval_test_{idx}"}}
            
            # 3. ACTUAL LANGGRAPH INVOCATION
            # We pass the question as a HumanMessage to the graph
            inputs = {"messages": [HumanMessage(content=question)]}
            
            # Run the graph
            response = graph_app.invoke(inputs, config=config)
            # ADD THIS: Wait between requests to stay under the per-minute limit
            print("Pausing to respect rate limits...")
            time.sleep(5)  # Start with 5 seconds; increase to 10 if you still get 429s
            
            # Extract the final AI output from the messages state
            ai_output = response["messages"][-1].content
            
            print(f" AI Answer: {ai_output[:100]}...\n")
            
        except Exception as e:
            print(f" [ERROR] Graph failed on this question: {e}\n")
            ai_output = f"ERROR: {str(e)}"

        answers.append(ai_output)
        
    # 4. Add the generated answers as a new column
    df['answer'] = answers
    
    # 5. Save to a new ready-to-evaluate CSV
    ready_csv_path = "evaluation/dataset_ready_for_eval.csv"
    df.to_csv(ready_csv_path, index=False)
    print(f"Success! Saved {len(answers)} real AI answers to {ready_csv_path}")
    print("You can now run 'python evaluation/runeval.py'")

if __name__ == "__main__":
    generate_pipeline_answers()