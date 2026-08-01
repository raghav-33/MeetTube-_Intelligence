
import yt_dlp
from pydub import AudioSegment
import os

# Download Directory 
DOWNLOAD_DIR = '.downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

# Download Audio from yt Video 
def download_youtube_audio(url :str) ->str:
    """Downloads the best quality audio from a YouTube URL and converts it to WAV."""
    
    # Define the output template (filename and extension placeholder) 
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",  # wav Format Downloading 
                "preferredquality": "192",
            }
        ],
        "quiet": True, # Suppresses massive terminal logs
        
        
        # ─── ADDED TO FIX YouTube blocks the request from Streamlit Cloud's IP address or HTTP 403 FORBIDDEN ERROR ───
        "cachedir": False,
        "nocheckcertificate": True,
        "rm_cachedir": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
        "extractor_args": {
            "youtube": {
                # Uses embedded TV & mobile web clients which bypass cloud datacenter IP blocks
                "player_client": ["tvembed", "mweb", "ios"],
            }
        }
    }
    
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extracts metadata and downloads the file
        info = ydl.extract_info(url, download=True)
        # yt-dlp might download a webm or m4a temporarily, this ensures our variable tracks the final .wav file
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename



# Converting into wav Enforcement , if download any wrong format
''' Speech-to-text APIs (like Whisper) perform best when audio is in a specific format (16kHz, single-channel WAV) and will crash if files are too large. '''

def convert_to_wav(input_path: str) -> str:
    """Converts local files to a strict 16kHz Mono WAV format for optimized AI transcription."""
    # Creates a new filename for the converted file
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path) #  Load the full audio file
    
    # STANDARDIZATION STEP:
    # set_channels(1) = Mono audio (left/right channels merged). 
    # set_frame_rate(16000) = 16kHz sample rate. This is exactly what Whisper is trained on.
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz , setting channel 1 : mono channel
    audio.export(output_path, format="wav")
    return output_path


# chunking the Whole Audio into 10 min per parts
def chunk_audio(wav_path : str , chunk_minutes : int = 2) -> list:
    audio = AudioSegment.from_wav(wav_path) # load full Audio file (in Milliseconds time format)
    chunk_ms = chunk_minutes * 60 * 1000  # # Define chunk length in milliseconds (3 minutes = 600,000 ms)

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]  # Slicing / cutting whole audio into 10 mins per part
        chunk_path = f"{wav_path}_chunk_{i}.wav"     # Save file as Chunk 1 , chunk 2 (i : indicate index / chunk no)
        chunk.export(chunk_path , format = "wav")   # # Export the chunk

        chunks.append(chunk_path)     
    
    return chunks

def process_input(source: str) -> list:
    """Main routing function: Determines the input type, standardizes it, and chunks it."""
    
    # Route 1: It's a web link
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        wav_path = convert_to_wav(wav_path)
        
    # Route 2: It's a local file upload (e.g., from Streamlit)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
    
    
    