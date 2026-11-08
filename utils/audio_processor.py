import os
import base64
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
MAX_CHUNK_MB = 23.0

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    # Setup browser impersonation and JS runtime args (switched to Node.js)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "force_ipv4": True,
        "impersonate": "chrome",  # Bypasses TLS fingerprinting
        "js_runtimes": {"node": {}},  # Tells yt-dlp to use Node.js
        "extractor_args": {
            "youtube": {
                "player_client": ["tv", "mweb", "web_safari"], # Bypasses bot checks
            }
        }
    }

    # Handle Cookies (if you added them to Streamlit Secrets)
    cookie_path = "/tmp/yt-dlp-cookies.txt"
    if os.getenv("YTDLP_COOKIES_B64"):
        try:
            decoded = base64.b64decode(os.getenv("YTDLP_COOKIES_B64")).decode('utf-8')
            with open(cookie_path, "w") as f:
                f.write(decoded)
            ydl_opts["cookiefile"] = cookie_path
            print("Loaded Base64 cookies.")
        except Exception as e:
            print(f"Failed to decode cookies: {e}")
    elif os.getenv("YTDLP_COOKIES_CONTENT"):
        with open(cookie_path, "w") as f:
            f.write(os.getenv("YTDLP_COOKIES_CONTENT"))
        ydl_opts["cookiefile"] = cookie_path
        print("Loaded raw text cookies.")

    # Download
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
        if os.path.exists(filename):
            return filename
        else:
            raise FileNotFoundError(f"Downloaded audio not found at {filename}")

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV (pydub)."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        if size_mb > MAX_CHUNK_MB:
            os.remove(chunk_path)
            mid = len(chunk) // 2
            for j, sub in enumerate([chunk[:mid], chunk[mid:]]):
                sub_path = f"{wav_path}_chunk_{i}_{j}.wav"
                sub.export(sub_path, format="wav")
                chunks.append(sub_path)
        else:
            chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected URL. Downloading audio...")
        raw_path = download_youtube_audio(source)
        print("Normalizing audio (16 kHz mono)...")
        wav_path = convert_to_wav(raw_path)
    else:
        print(f"Detected local file: {source}. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
