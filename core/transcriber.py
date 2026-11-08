import os
import time
from dotenv import load_dotenv

# Load .env BEFORE any os.getenv() call
load_dotenv()

from groq import Groq

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _transliterate
    _HAS_TRANSLIT = True
except ImportError:
    _HAS_TRANSLIT = False
    print("WARNING: indic-transliteration not installed. Hinglish mode will "
          "fall back to raw Devanagari transcript.")

# Default to Groq's whisper model
_API_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-large-v3")
_MAX_UPLOAD_MB = 24.0

_client = None

def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from your .env file.")
    return Groq(api_key=api_key)


# ─── Core call ─────────────────────────────────────────────────────────────────
def _transcribe_one(path: str, language_code: str | None) -> str:
    client = _get_client()
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=_API_MODEL,  # <-- Uses the variable now
            file=(os.path.basename(path), f),
            language=language_code,
            response_format="text",
        )
    return resp


def _transcribe_with_retry(path: str, language_code: str | None,
                           max_attempts: int = 3) -> str:
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _transcribe_one(path, language_code)
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(t in msg for t in ("429", "rate", "timeout", "503", "502")):
                wait = 2 ** attempt
                print(f"Transcription attempt {attempt} failed ({e}); "
                      f"retrying in {wait}s ...")
                time.sleep(wait)
                continue
            raise
    raise last


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return _transcribe_with_retry(chunk_path, "hi")
    return _transcribe_with_retry(chunk_path, "en")


def devanagari_to_hinglish(text: str) -> str:
    if not _HAS_TRANSLIT or not text.strip():
        return text
    try:
        return _transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception as e:
        print(f"Transliteration failed ({e}); keeping Devanagari.")
        return text


def transcribe_all(chunks: list, language: str = "english",
                   progress_callback=None) -> str:
    full_transcript = ""
    
    # FIXED: Accurately reflect the engine being used
    engine = f"Groq API → {_API_MODEL}"
    lang_tag = "Hindi→Hinglish" if language.lower() == "hinglish" else "English"
    print(f"Using {engine} ({lang_tag}) for transcription.")

    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total, "transcribing")
        print(f"Transcribing chunk {i + 1}/{total} via API ...")
        full_transcript += transcribe_chunk(chunk, language) + " "
        if progress_callback:
            progress_callback(i + 1, total, "done")

    transcript = full_transcript.strip()

    if language.lower() == "hinglish":
        if progress_callback:
            progress_callback(total, total, "transliterating")
        print("Transliterating Devanagari → Hinglish ...")
        transcript = devanagari_to_hinglish(transcript)

    return transcript