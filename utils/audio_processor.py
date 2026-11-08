import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Groq free tier rejects >25 MB. 10-min @ 16 kHz mono ≈ 18.3 MiB → safe.
MAX_CHUNK_MB = 23.0


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Downloaded audio not found at {filename}")
    return filename


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
            # Rare safety net (e.g. custom sample rate). Split in half.
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
        print("Detected YouTube URL. Downloading audio...")
        raw_path = download_youtube_audio(source)
        # CRITICAL: normalize to 16 kHz mono so API uploads stay < 25 MB.
        print("Normalizing downloaded audio (16 kHz mono)...")
        wav_path = convert_to_wav(raw_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks