"""
ui/app.py
Run: streamlit run app.py

Requirements:
  - ollama serve  (keep running in background)
  - ollama pull mistral
  - ffmpeg installed and on PATH  (for video audio extraction)
  - tesseract installed            (for image/frame OCR)
"""

import sys, os, shutil, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import ollama  # type: ignore
from ingestion.ingestor      import ingest_file
from embeddings.vector_store import VectorStore
from retrieval.query_engine  import QueryEngine


st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{
  --bg:#0d0f14; --surf:#161920; --surf2:#1e2130; --border:#2a2f42;
  --accent:#6c63ff; --aglow:rgba(108,99,255,0.15); --accent2:#ff6b9d;
  --t1:#f0f2ff; --t2:#9ba3c4; --t3:#5c6285; --green:#3ddc97;
}
html,body,.stApp{background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--t1)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 6rem!important;max-width:880px!important;margin:0 auto!important;}
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"]>div:first-child{padding:24px 18px!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] div{color:var(--t2)!important;}
[data-testid="stSelectbox"]>div>div{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--t1)!important;}
[data-testid="stFileUploader"]{background:var(--surf2)!important;border:1.5px dashed var(--border)!important;border-radius:12px!important;}
[data-testid="stFileUploader"] *{color:var(--t2)!important;}
.stButton>button{background:var(--surf2)!important;border:1px solid var(--border)!important;color:var(--t1)!important;border-radius:10px!important;font-family:'DM Sans',sans-serif!important;font-size:0.84rem!important;font-weight:500!important;transition:all 0.18s!important;}
.stButton>button:hover{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--aglow)!important;transform:translateY(-1px)!important;}
.stProgress>div>div{background:var(--accent)!important;border-radius:99px!important;}
.stProgress>div{background:var(--surf2)!important;border-radius:99px!important;}
[data-testid="stChatInput"]{background:var(--surf)!important;border-top:1px solid var(--border)!important;}
[data-testid="stChatInput"] textarea{background:var(--surf2)!important;border:1.5px solid var(--border)!important;border-radius:14px!important;color:var(--t1)!important;font-family:'DM Sans',sans-serif!important;font-size:0.95rem!important;caret-color:var(--accent)!important;}
[data-testid="stChatInput"] textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--aglow)!important;outline:none!important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--t3)!important;}
[data-testid="stChatInput"] button{background:var(--accent)!important;border-radius:10px!important;border:none!important;}
[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:4px 0!important;}
.streamlit-expanderHeader{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--t2)!important;font-size:0.82rem!important;}
.streamlit-expanderContent{background:var(--surf2)!important;border:1px solid var(--border)!important;border-top:none!important;border-radius:0 0 10px 10px!important;}
hr{border-color:var(--border)!important;}
[data-testid="stAlert"]{background:var(--surf2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--t2)!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}
.hero{display:flex;align-items:center;gap:14px;padding-bottom:20px;border-bottom:1px solid var(--border);margin-bottom:6px;}
.hero-icon{width:46px;height:46px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 18px var(--aglow);flex-shrink:0;}
.hero-title{font-family:'Syne',sans-serif;font-size:1.75rem;font-weight:800;color:var(--t1)!important;letter-spacing:-0.4px;margin:0;line-height:1;}
.hero-sub{font-size:0.81rem;color:var(--t3)!important;margin:5px 0 0;}
.pills{display:flex;gap:7px;flex-wrap:wrap;margin:18px 0 28px;}
.pill{background:var(--surf);border:1px solid var(--border);border-radius:99px;padding:4px 13px;font-size:0.74rem;color:var(--t2)!important;display:inline-flex;align-items:center;gap:5px;}
.pill-dot{width:5px;height:5px;border-radius:50%;display:inline-block;}
.user-bubble{background:linear-gradient(135deg,#1e2248,#1a1d38);border:1px solid #2e3460;border-radius:18px 18px 4px 18px;padding:12px 17px;color:#f0f2ff!important;font-size:0.95rem;line-height:1.65;display:inline-block;max-width:100%;word-break:break-word;}
.answer-card{background:var(--surf);border:1px solid var(--border);border-radius:4px 18px 18px 18px;padding:18px 22px 18px 26px;color:#f0f2ff!important;font-size:0.95rem;line-height:1.78;margin-bottom:10px;position:relative;}
.answer-card::before{content:'';position:absolute;top:0;bottom:0;left:0;width:3px;background:linear-gradient(180deg,var(--accent),var(--accent2));border-radius:3px 0 0 3px;}
.cites{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.cite-tag{background:var(--surf2);border:1px solid var(--border);border-radius:7px;padding:4px 11px;font-size:0.74rem;color:var(--t2)!important;}
.s-label{font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:var(--t3)!important;margin:18px 0 8px;display:block;}
.f-row{display:flex;align-items:center;gap:7px;padding:7px 0;border-bottom:1px solid var(--border);font-size:0.8rem;color:var(--t2)!important;overflow:hidden;}
.f-badge{font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:5px;flex-shrink:0;}
.b-pdf {background:rgba(255,107,107,.15);color:#ff6b6b!important;border:1px solid rgba(255,107,107,.3);}
.b-img {background:rgba(255,181,71,.12); color:#ffb547!important;border:1px solid rgba(255,181,71,.3);}
.b-vid {background:rgba(108,99,255,.15); color:#a29bfe!important;border:1px solid rgba(108,99,255,.3);}
.chunk-card{background:var(--bg);border:1px solid var(--border);border-radius:9px;padding:11px 13px;margin:5px 0;font-size:0.79rem;color:var(--t2)!important;line-height:1.55;}
.chunk-card strong{color:var(--t1)!important;}
.s-tag{display:inline-block;background:rgba(108,99,255,.15);color:var(--accent)!important;border:1px solid rgba(108,99,255,.3);border-radius:5px;padding:1px 7px;font-size:0.7rem;font-weight:600;margin-left:5px;}
.empty{text-align:center;padding:70px 20px;}
.empty-icon{font-size:3.2rem;margin-bottom:14px;}
.empty h3{font-family:'Syne',sans-serif;font-size:1.1rem;color:var(--t2)!important;margin:0 0 5px;}
.empty p{font-size:0.83rem;color:var(--t3)!important;margin:0;}
.stMarkdown p,.stMarkdown span,.element-container p,.element-container span,p,li{color:var(--t1)!important;}
</style>
""", unsafe_allow_html=True)


# ── File type sets ────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


# ── Diagnostics ───────────────────────────────────────

def _model_names(models_response) -> list:
    names = []
    for m in models_response.get("models", []):
        if isinstance(m, dict):
            names.append(m.get("name", "") or m.get("model", ""))
        else:
            names.append(getattr(m, "model", None) or getattr(m, "name", ""))
    return names


def run_diagnostics():
    results = {}

    # Ollama server
    try:
        ollama.list()
        results["Ollama server"] = (True, "running ✓")
    except Exception:
        results["Ollama server"] = (False, "NOT running → open terminal: ollama serve")

    # Mistral model
    try:
        available = _model_names(ollama.list())
        found = any("mistral" in m for m in available)
        results["Mistral model"] = (found, "ready ✓" if found else "run: ollama pull mistral")
    except Exception:
        results["Mistral model"] = (False, "start Ollama first")

    # PyMuPDF
    try:
        import fitz  # type: ignore
        results["PDF  (PyMuPDF)"] = (True, "installed ✓")
    except ImportError:
        results["PDF  (PyMuPDF)"] = (False, "pip install PyMuPDF")

    # Pillow
    try:
        from PIL import Image
        results["IMG  (Pillow)"] = (True, "installed ✓")
    except ImportError:
        results["IMG  (Pillow)"] = (False, "pip install Pillow")

    # pytesseract binding
    try:
        import pytesseract  # type: ignore
        results["IMG  (pytesseract)"] = (True, "installed ✓")
    except ImportError:
        results["IMG  (pytesseract)"] = (False, "pip install pytesseract")

    # Tesseract binary
    try:
        import pytesseract  # type: ignore
        pytesseract.get_tesseract_version()
        results["IMG  (Tesseract binary)"] = (True, "found ✓")
    except Exception:
        results["IMG  (Tesseract binary)"] = (False,
            "Install Tesseract:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  Linux:   sudo apt install tesseract-ocr\n"
            "  Mac:     brew install tesseract")

    # OpenCV
    try:
        import cv2  # type: ignore
        results["VID  (opencv-python)"] = (True, "installed ✓")
    except ImportError:
        results["VID  (opencv-python)"] = (False, "pip install opencv-python")

    # ffmpeg binary
    results["VID  (ffmpeg binary)"] = (
        shutil.which("ffmpeg") is not None,
        "found ✓" if shutil.which("ffmpeg") else
        "Install ffmpeg:\n"
        "  Windows: https://ffmpeg.org/download.html\n"
        "  Linux:   sudo apt install ffmpeg\n"
        "  Mac:     brew install ffmpeg"
    )

    # openai-whisper
    try:
        import whisper  # type: ignore
        results["VID  (whisper)"] = (True, "installed ✓")
    except ImportError:
        results["VID  (whisper)"] = (False, "pip install openai-whisper")

    return results


# ── Session state ─────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vs" not in st.session_state:
    st.session_state.vs = VectorStore()

if "qe" not in st.session_state:
    st.session_state.qe = QueryEngine(st.session_state.vs)

vs = st.session_state.vs
qe = st.session_state.qe


# ── Sidebar ───────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<span class="s-label">⚙️ Settings</span>',
        unsafe_allow_html=True
    )

    top_k = st.selectbox(
        "Results per query",
        options=[3, 5, 7, 10],
        index=1,
        key="top_k"
    )

    st.divider()

    # Diagnostics
    with st.expander("🔍 System diagnostics"):
        diag = run_diagnostics()
        for name, (ok, msg) in diag.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"**{icon} {name}**  \n`{msg}`")

    st.divider()

    # Upload
    st.markdown('<span class="s-label">📤 Upload files</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "PDF, Images, or Video",
        type=["pdf",
              "png", "jpg", "jpeg", "tiff", "bmp", "webp",
              "mp4", "mov", "avi", "mkv", "webm", "flv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded and st.button("⚡ Index files", use_container_width=True):
        status        = st.empty()
        prog          = st.progress(0)
        total_indexed = 0

        for i, uf in enumerate(uploaded):
            status.info(f"Processing {uf.name}…")

            # Save to temp file preserving extension
            ext = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uf.read())
                fp = tmp.name

            try:
                records = ingest_file(fp)
                if not records:
                    st.warning(f"⚠️ {uf.name} — no text extracted.")
                else:
                    vs.add_records(records)
                    total_indexed += len(records)
                    st.success(f"✓ {uf.name} — {len(records)} chunks indexed")
            except EnvironmentError as e:
                st.error(f"❌ {uf.name}\n{e}")
            except ImportError as e:
                st.error(f"❌ {uf.name} — missing library:\n{e}")
            except RuntimeError as e:
                st.error(f"❌ {uf.name} — failed:\n{e}")
            except Exception as e:
                st.error(f"❌ {uf.name} — {type(e).__name__}: {e}")
            finally:
                os.unlink(fp)

            prog.progress((i + 1) / len(uploaded))

        if total_indexed > 0:
            status.success(f"✅ Done! {total_indexed} chunks indexed.")
        else:
            status.warning("⚠️ No new chunks were indexed.")

        st.rerun()

    st.divider()

    # Knowledge base
    st.markdown('<span class="s-label">Knowledge base</span>', unsafe_allow_html=True)
    files = vs.list_sources()

    if files:
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".pdf":
                badge = '<span class="f-badge b-pdf">PDF</span>'
            elif ext in IMAGE_EXTS:
                badge = '<span class="f-badge b-img">IMG</span>'
            else:
                badge = '<span class="f-badge b-vid">VID</span>'

            c1, c2 = st.columns([5, 1])
            short  = fname if len(fname) <= 20 else fname[:18] + "…"
            c1.markdown(f'<div class="f-row">{badge} {short}</div>', unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{fname}"):
                vs.delete_source(fname)
                st.rerun()
    else:
        st.markdown(
            '<p style="font-size:0.78rem;color:#5c6285!important;">No files indexed yet.</p>',
            unsafe_allow_html=True
        )

    st.divider()
    n = vs.total_chunks()
    st.markdown(
        f'<p style="font-size:0.76rem;color:#5c6285!important;">'
        f'<span style="color:#6c63ff!important;font-weight:600;">{n}</span>'
        f' chunks indexed</p>',
        unsafe_allow_html=True
    )


# ── MAIN ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-icon">🧠</div>
  <div>
    <div class="hero-title">DocMind</div>
    <div class="hero-sub">Ask questions across PDF, Images &amp; Video — powered by Mistral</div>
  </div>
</div>
<div class="pills">
  <span class="pill"><span class="pill-dot" style="background:#ff6b6b"></span>PDF · Page indexed</span>
  <span class="pill"><span class="pill-dot" style="background:#ffb547"></span>Images · OCR indexed</span>
  <span class="pill"><span class="pill-dot" style="background:#a29bfe"></span>Video · Frame OCR + Audio transcribed</span>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.chat_history:
    if vs.total_chunks() == 0:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">📂</div>
          <h3>No documents uploaded</h3>
          <p>Upload PDF, images, or video files in the sidebar to get started.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">💬</div>
          <h3>Ready to answer</h3>
          <p>Type a question below to search your documents.</p>
        </div>""", unsafe_allow_html=True)

# Chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f'<div class="user-bubble">{entry["question"]}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f'<div class="answer-card">{entry["answer"]}</div>', unsafe_allow_html=True)
        if entry.get("citations"):
            tags = "".join(f'<span class="cite-tag">{c}</span>' for c in entry["citations"])
            st.markdown(f'<div class="cites">{tags}</div>', unsafe_allow_html=True)

# Chat input
question = st.chat_input("Ask anything about your documents…")

if question:
    if vs.total_chunks() == 0:
        st.warning("⚠️ Please upload and index files first.")
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                result = qe.ask(question, top_k=top_k)

            st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

            if result.get("citations"):
                tags = "".join(f'<span class="cite-tag">{c}</span>' for c in result["citations"])
                st.markdown(f'<div class="cites">{tags}</div>', unsafe_allow_html=True)

            with st.expander("View retrieved chunks"):
                for ch in result["chunks"]:
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>{ch["citation"]}</strong>'
                        f'<span class="s-tag">Score {ch["score"]}</span>'
                        f'<br><br>{ch["text"][:420]}…'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.session_state.chat_history.append({
            "question":  question,
            "answer":    result["answer"],
            "citations": result["citations"],
        })

# Clear chat
if st.session_state.chat_history:
    st.divider()
    if st.button("🗑️  Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()