# AI Video Assistant (Meeting Intelligence) 🎬

**🔴 [Live Demo on Streamlit](https://ai-video-meeting-assistant-dmvyxwrubaziueuv3byqxe.streamlit.app/)**

An end-to-end RAG (Retrieval-Augmented Generation) application that transforms long-form audio and videos into actionable insights. It transcribes audio, generates map-reduce summaries, extracts key decisions, and allows users to chat directly with the meeting transcript.

## 🧠 Architecture & Features

- **Audio Transcription:** Utilizes Groq's `whisper-large-v3` API for ultra-fast, highly accurate speech-to-text processing.
- **Map-Reduce Summarization:** Splits long transcripts into chunks, summarizes them using Google Gemini, and combines them into a cohesive final summary.
- **RAG Chat Interface:** Embeds the transcript into ChromaDB using Gemini embeddings, allowing users to ask natural language questions and get context-aware answers.
- **Hinglish Support:** Features custom prompt engineering and Devanagari-to-Latin transliteration, allowing seamless processing of Hindi/Urdu mixed with English.
- **Robust File Processing:** Includes an audio/video file uploader and custom chunking logic to bypass API size limits and cloud IP restrictions (e.g., YouTube datacenter blocks).

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom CSS for dark-mode UI)
- **LLM & Orchestration:** LangChain, Google Gemini (Chat & Embeddings)
- **Transcription API:** Groq (Whisper-large-v3)
- **Vector Database:** ChromaDB
- **Audio Processing:** FFmpeg, Pydub, yt-dlp
- **Utilities:** Indic-Transliteration, Python-dotenv

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarfraz-ali-ai/ai-video-meeting-assistant.git
   cd ai-video-meeting-assistant
