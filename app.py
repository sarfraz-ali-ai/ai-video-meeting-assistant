import streamlit as st
import time
import os
import traceback
import html as _html
import re as _re
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f; --surface: #111118; --surface-2: #1a1a25; --border: #2a2a3a;
    --accent: #7c3aed; --accent-glow: #9f67ff; --accent-2: #06b6d4;
    --text: #e8e8f0; --text-muted: #7070a0;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
}

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800; line-height: 1.1; margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.5rem;
}

.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1rem; position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.card-title {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;
}
.card-content { font-size: 0.875rem; line-height: 1.7; color: var(--text); }
.card-content strong { color: var(--accent-glow); }
.card-content em     { color: var(--accent-2); }

.badge {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.875rem !important; letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important; transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
}

.status-bar {
    display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem;
    background: var(--surface-2); border-radius: 8px; margin: 0.4rem 0;
    border: 1px solid var(--border); font-size: 0.8rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

.chat-container {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.25rem; max-height: 420px; overflow-y: auto; margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble {
    display: inline-block; padding: 0.6rem 1rem; border-radius: 10px;
    font-size: 0.85rem; line-height: 1.6; max-width: 90%;
}
.chat-bubble strong { color: #fff; }
.chat-bubble em     { color: var(--accent-2); }
.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }
.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.transcript-box {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.25rem; font-size: 0.82rem; line-height: 1.8; max-height: 300px;
    overflow-y: auto; color: var(--text-muted); white-space: pre-wrap; word-break: break-word;
}

.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── HTML-safe markdown helper ──────────────────────────────────────────────────
def _md_to_safe_html(text) -> str:
    if text is None:
        return ""
    text = _html.escape(str(text))
    text = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=_re.S)
    text = _re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>",
                   text, flags=_re.S)
    text = text.replace("\n", "<br>")
    return text

# ─── Pre-flight check ──────────────────────────────────────────────────────────
def _preflight_check(language: str) -> list:
    issues = []
    if not os.getenv("GROQ_API_KEY"):
        issues.append("❌ `GROQ_API_KEY` is missing. Transcription will fail.")
    if not os.getenv("GOOGLE_API_KEY"):
        issues.append("❌ `GOOGLE_API_KEY` is missing. LLM and Embeddings will fail.")

    if language.lower() == "hinglish":
        try:
            import indic_transliteration  # noqa: F401
        except ImportError:
            issues.append("❌ `indic-transliteration` not installed — Hinglish output will fail.")

    return issues

# ─── Session State Init ────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    
    # 📂 FILE UPLOADER (Reliable Fallback)
    uploaded_file = st.file_uploader("Upload Audio/Video (Recommended)", type=["mp3", "wav", "mp4", "m4a"])
    
    # 🔗 YOUTUBE URL (If not blocked)
    source = st.text_input("Or paste YouTube URL", placeholder="https://youtube.com/watch?v=...")

    # Handle Uploaded File
    if uploaded_file is not None:
        os.makedirs("downloades", exist_ok=True)
        source = os.path.join("downloades", uploaded_file.name)
        with open(source, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File ready: {uploaded_file.name}")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please upload a file or enter a YouTube URL.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        issues = _preflight_check(language)
        if issues:
            with st.expander("⚠️ Pre-flight notices", expanded=True):
                for i in issues:
                    st.warning(i)

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with st.status("⚙️ Running pipeline…", expanded=True) as status:
                # 1. Audio
                t0 = time.time()
                update_step("audio", "active")
                st.write("🔊 **[1/6]** Processing audio…")
                chunks = process_input(source)
                update_step("audio", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ {len(chunks)} chunk(s) ready ({time.time()-t0:.1f}s)")

                # 2. Transcribe
                t0 = time.time()
                update_step("transcript", "active")
                st.write(f"📝 **[2/6]** Transcribing {len(chunks)} chunk(s) via Groq…")
                progress_bar = st.progress(0, text="⏳ Preparing transcription…")
                
                def _prog(done, total, phase):
                    pct = int(done / total * 100) if total else 0
                    progress_bar.progress(pct, text=f"⏳ Chunk {done}/{total} transcribing…")

                transcript = transcribe_all(chunks, language, progress_callback=_prog)
                progress_bar.empty()
                update_step("transcript", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ {len(transcript)} chars transcribed ({time.time()-t0:.1f}s)")

                # 3. Title
                t0 = time.time()
                update_step("title", "active")
                st.write("🏷️ **[3/6]** Generating title via Gemini…")
                title = generate_title(transcript)
                update_step("title", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ “{title}” ({time.time()-t0:.1f}s)")

                # 4. Summary
                t0 = time.time()
                update_step("summary", "active")
                st.write("📋 **[4/6]** Summarising (chunked map-reduce)…")
                summary = summarize(transcript, language)
                update_step("summary", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ Summary ready ({time.time()-t0:.1f}s)")

                # 5. Extraction
                t0 = time.time()
                update_step("extract", "active")
                st.write("🔍 **[5/6]** Extracting action items → decisions → questions…")
                action_items = extract_action_items(transcript, language)
                decisions = extract_key_decisions(transcript, language)
                questions = extract_questions(transcript, language)
                update_step("extract", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ Extraction complete ({time.time()-t0:.1f}s)")

                # 6. RAG
                t0 = time.time()
                update_step("rag", "active")
                st.write("🧠 **[6/6]** Building RAG index…")
                rag_chain = build_rag_chain(transcript, language)
                update_step("rag", "done")
                st.write(f"&nbsp;&nbsp;&nbsp;✅ RAG ready ({time.time()-t0:.1f}s)")

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            time.sleep(0.3)
            st.rerun()

        except Exception as e:
            traceback.print_exc()
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            st.error(f"❌ Pipeline failed: {e}")
            with st.expander("📋 Full traceback (copy for debugging)"):
                st.code(traceback.format_exc())

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {_md_to_safe_html(r['title'])}
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{_md_to_safe_html(r['summary'])}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{_md_to_safe_html(r["transcript"])}</div>',
                unsafe_allow_html=True,
            )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{_md_to_safe_html(r['action_items'])}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{_md_to_safe_html(r['key_decisions'])}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{_md_to_safe_html(r['open_questions'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            safe = _md_to_safe_html(msg["content"])
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{safe}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{safe}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Upload an audio/video file or paste a YouTube URL in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)
