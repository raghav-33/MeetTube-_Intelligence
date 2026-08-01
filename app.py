import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# NEW IMPORT ADDED HERE TO FIX THE DOUBLE PRINTING
from langchain_core.messages import AIMessageChunk

# ─── 1. IMPORT YOUR ARCHITECTURE MODULES ───
from agents.graph import graph_app
from utils.audio_processor import process_input
from core.transcriber import transcribe_chunks_api
from RAG.rag import AdvancedHybridRAG

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide"
)

# ─── 1. CACHED RAG INSTANCE INITIALIZATION ───
@st.cache_resource
def get_rag_store():
    return AdvancedHybridRAG()

# Initialize your RAG system instance 
if "rag_store" not in st.session_state:
    st.session_state.rag_store = get_rag_store()

# ─── 2. DYNAMIC THREAD & CHAT HISTORY MANAGEMENT ───
# We use a dictionary to store multiple ChatGPT-like sessions.
if "chats" not in st.session_state:
    # Create the very first default thread
    initial_thread_id = str(uuid.uuid4())
    st.session_state.chats = {
        initial_thread_id: {
            "title": "New Conversation",
            "messages": [],
            "transcript": "" # Each chat remembers its own video transcript
        }
    }
    st.session_state.current_thread_id = initial_thread_id

# Helper variable for the active chat
active_chat = st.session_state.chats[st.session_state.current_thread_id]

# LangGraph config bound to the active thread
config = {"configurable": {"thread_id": st.session_state.current_thread_id}}


# ─── 3. SIDEBAR: CHAT HISTORY & INGESTION PIPELINE ───
with st.sidebar:
    st.header("💬 Chat History")
    
    # NEW CHAT BUTTON
    if st.button("➕ New Chat", use_container_width=True):
        new_thread_id = str(uuid.uuid4())
        st.session_state.chats[new_thread_id] = {
            "title": f"Chat {len(st.session_state.chats) + 1}",
            "messages": [],
            "transcript": ""
        }
        st.session_state.current_thread_id = new_thread_id
        st.rerun() # Refresh UI immediately

    st.divider()
    
    # LIST PREVIOUS CHATS
    st.subheader("Previous Conversations")
    for thread_id, chat_data in st.session_state.chats.items():
        # Highlight the currently active chat
        button_type = "primary" if thread_id == st.session_state.current_thread_id else "secondary"
        if st.button(chat_data["title"], key=thread_id, type=button_type, use_container_width=True):
            st.session_state.current_thread_id = thread_id
            st.rerun()

    st.divider()
    
    # VIDEO INGESTION SECTION
    st.header("📥 Upload Media")
    input_type = st.radio("Choose Input Type:", ["YouTube URL", "Upload Meeting Recording"])
    
    if input_type == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube Link:")
        if st.button("Process Video", use_container_width=True):
            if youtube_url.strip() == "":
                st.warning("Please enter a valid YouTube URL first.")
            else:
                with st.spinner("Downloading, chunking, and transcribing..."):
                    try:
                        st.info("Downloading and processing audio chunks...")
                        audio_chunks = process_input(youtube_url)
                        
                        st.info("Transcribing audio chunks via Groq...")
                        full_transcript = transcribe_chunks_api(audio_chunks)
                        with open("transcript.txt", "w", encoding="utf-8") as f:
                            f.write(full_transcript)
                        
                        # Save transcript STRICTLY to the current active chat thread
                        active_chat["transcript"] = full_transcript
                        
                        st.info("Generating embeddings and indexing to database...")
                        st.session_state.rag_store.ingest_transcript(full_transcript)
                        
                        # Update the chat title dynamically based on processing
                        active_chat["title"] = "YouTube Video Chat"
                        st.success("Video successfully processed and indexed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
                
    else:
        uploaded_file = st.file_uploader("Upload an audio/video file", type=["mp3", "mp4", "wav"])
        if uploaded_file and st.button("Process Local File", use_container_width=True):
            with st.spinner("Processing local file data..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.info("Slicing audio file segments...")
                    audio_chunks = process_input(temp_path)
                    
                    st.info("Transcribing audio chunks via Groq...")
                    full_transcript = transcribe_chunks_api(audio_chunks)
                    with open("transcript.txt", "w", encoding="utf-8") as f:
                        f.write(full_transcript)
                    
                    # Save transcript STRICTLY to the current active chat thread
                    active_chat["transcript"] = full_transcript
                    
                    st.info("Generating embeddings and indexing to database...")
                    st.session_state.rag_store.ingest_transcript(full_transcript)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                    active_chat["title"] = "Local File Chat"
                    st.success("File successfully processed and indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ─── 4. MAIN CHAT INTERFACE ───
st.title("🎥 AI Video Assistant")
st.caption(f"Currently viewing: **{active_chat['title']}**")

# Render conversation history for the CURRENT thread only
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ─── 5. USER CHAT INTERACTION & FILTERED STREAMING ───
if user_query := st.chat_input("Ask something about the video context or chat directly:"):
    
    # 1. Instantly display user message
    with st.chat_message("user"):
        st.write(user_query)
        
    # 2. Append to current thread's state
    active_chat["messages"].append({"role": "user", "content": user_query})
    
    # 3. Rename chat title if it's the first message
    if active_chat["title"] == "New Conversation":
        # Create a short title from the first query (first 20 chars)
        active_chat["title"] = user_query[:20] + "..." if len(user_query) > 20 else user_query
    
    # 4. Pass user query AND the specific thread's transcript to LangGraph
    state_input = {
        "messages": [("user", user_query)],
        "transcript": active_chat.get("transcript", "")
    }
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk, metadata in graph_app.stream(
                state_input,
                config=config,
                stream_mode="messages"
            ):
                target_nodes = ["direct_chat_executor", "rag_executor", "summary_executor"]
                
                # FIX: We now strictly verify that the chunk is an AIMessageChunk.
                # This guarantees we only render the streaming tokens and ignore the final duplicate message.
                if metadata.get("langgraph_node") in target_nodes:
                    if isinstance(chunk, AIMessageChunk) and hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response.strip())
            
            # 5. Append assistant response to current thread's state
            active_chat["messages"].append({"role": "assistant", "content": full_response.strip()})
            
        except Exception as e:
            st.error(f"Pipeline Error encountered during graph execution: {e}")