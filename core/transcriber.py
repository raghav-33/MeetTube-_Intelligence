import os
import httpx
from groq import Groq

def transcribe_chunks_api(chunk_paths: list, chunk_minutes: int = 2) -> str:
    """Transcribes audio and embeds calculated timestamps for RAG citations."""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GROQ_API_KEY is missing! Ensure your .env file is loaded.")

    http_client = httpx.Client(verify=False, timeout=300.0)
    client = Groq(api_key=api_key, http_client=http_client)

    full_transcript = ""
    
    for idx, chunk_path in enumerate(chunk_paths):
        # Calculate the start and end time based on the chunk index
        start_min = idx * chunk_minutes
        end_min = (idx + 1) * chunk_minutes
        
        # Format as [MM:SS - MM:SS]
        timestamp_tag = f"[{start_min:02d}:00 - {end_min:02d}:00]"
        
        print(f"Loading chunk {idx + 1} {timestamp_tag} into RAM and sending to Groq...")
        
        with open(chunk_path, "rb") as f:
            audio_bytes = f.read()
            
        try:
            response = client.audio.translations.create(
                model="whisper-large-v3", 
                file=(os.path.basename(chunk_path), audio_bytes, "audio/wav")
            )
            
            # Inject the timestamp tag directly into the transcript text
            full_transcript += f"\n{timestamp_tag}\n{response.text.strip()}\n"
            
        except Exception as e:
            print(f"❌ Error transcribing chunk {idx}: {e}")
            raise e
        
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            
    return full_transcript.strip()