"""
ingestion/ingestor.py
Handles PDF, DOCX, Images, Audio.
Lazy imports — missing library only breaks that file type.
"""

import os
import shutil


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _rec(text, source, file_type, page, total_pages, chunk, rec_id, timestamp=""):
    return {
        "text":        text,
        "source":      source,
        "file_type":   file_type,
        "page":        int(page),
        "total_pages": int(total_pages),
        "chunk":       int(chunk),
        "timestamp":   str(timestamp),
        "id":          str(rec_id),
    }


# ── PDF ───────────────────────────────────────────────

def ingest_pdf(file_path: str):
    try:
        import fitz # type: ignore
    except ImportError:
        raise ImportError("Run: pip install PyMuPDF")

    records  = []
    filename = os.path.basename(file_path)
    doc      = fitz.open(file_path)

    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        if not text:
            continue
        for i, chunk in enumerate(split_into_chunks(text)):
            records.append(_rec(chunk, filename, "pdf",
                                page_num + 1, len(doc), i,
                                f"{filename}_p{page_num+1}_c{i}"))
    doc.close()
    print(f"[PDF] {filename} → {len(records)} chunks")
    return records


# ── DOCX ──────────────────────────────────────────────

def ingest_docx(file_path: str):
    try:
        from docx import Document # type: ignore
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    records            = []
    filename           = os.path.basename(file_path)
    doc                = Document(file_path)
    paragraphs         = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    paragraphs_per_sec = 10
    total_secs         = max(1, (len(paragraphs) + paragraphs_per_sec - 1) // paragraphs_per_sec)
    section_num        = 1

    for i in range(0, len(paragraphs), paragraphs_per_sec):
        section_text = " ".join(paragraphs[i : i + paragraphs_per_sec])
        for j, chunk in enumerate(split_into_chunks(section_text)):
            records.append(_rec(chunk, filename, "docx",
                                section_num, total_secs, j,
                                f"{filename}_s{section_num}_c{j}"))
        section_num += 1

    print(f"[DOCX] {filename} → {len(records)} chunks")
    return records


# ── IMAGE ─────────────────────────────────────────────

def ingest_image(file_path: str):
    try:
        import pytesseract # type: ignore
        from PIL import Image
    except ImportError:
        raise ImportError("Run: pip install pytesseract Pillow\nAlso install Tesseract binary.")

    filename = os.path.basename(file_path)
    records  = []

    try:
        image = Image.open(file_path)
        text  = pytesseract.image_to_string(image).strip()
    except Exception as e:
        raise RuntimeError(f"OCR failed for {filename}: {e}")

    if not text:
        print(f"[IMAGE] No text found in {filename}")
        return []

    for i, chunk in enumerate(split_into_chunks(text)):
        records.append(_rec(chunk, filename, "image", 1, 1, i,
                            f"{filename}_img_c{i}"))

    print(f"[IMAGE] {filename} → {len(records)} chunks")
    return records


# ── AUDIO ─────────────────────────────────────────────

_whisper_model = None


def ingest_audio(file_path: str):
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg not found.\n"
            "Windows: https://ffmpeg.org/download.html (add to PATH)\n"
            "Linux:   sudo apt install ffmpeg\n"
            "Mac:     brew install ffmpeg"
        )

    try:
        import whisper # type: ignore
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    global _whisper_model
    if _whisper_model is None:
        print("[Whisper] Loading model...")
        _whisper_model = whisper.load_model("base")
        print("[Whisper] Ready!")

    filename = os.path.basename(file_path)
    print(f"[Audio] Transcribing {filename}...")

    try:
        result = _whisper_model.transcribe(file_path, verbose=False, fp16=False)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

    segments = result.get("segments", [])
    if not segments:
        return []

    records      = []
    buffer       = ""
    buffer_start = 0.0
    first        = True
    idx          = 0

    def fmt_ts(s):
        m, sec = divmod(int(s), 60)
        h, m   = divmod(m, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    for seg in segments:
        t = seg.get("text", "").strip()
        if not t:
            continue
        if first:
            buffer_start = seg.get("start", 0.0)
            first = False
        buffer += " " + t

        if len(buffer.split()) >= 300:
            records.append(_rec(buffer.strip(), filename, "audio",
                                idx + 1, 0, idx,
                                f"{filename}_t{idx}",
                                timestamp=fmt_ts(buffer_start)))
            idx   += 1
            buffer = ""
            first  = True

    if buffer.strip():
        records.append(_rec(buffer.strip(), filename, "audio",
                            idx + 1, 0, idx,
                            f"{filename}_t{idx}",
                            timestamp=fmt_ts(buffer_start)))

    for r in records:
        r["total_pages"] = len(records)

    print(f"[Audio] {filename} → {len(records)} chunks")
    return records


# ── ROUTER ────────────────────────────────────────────

SUPPORTED = {
    ".pdf":  ingest_pdf,
    ".docx": ingest_docx,
    ".doc":  ingest_docx,
    ".png":  ingest_image,
    ".jpg":  ingest_image,
    ".jpeg": ingest_image,
    ".tiff": ingest_image,
    ".bmp":  ingest_image,
    ".mp3":  ingest_audio,
    ".wav":  ingest_audio,
    ".m4a":  ingest_audio,
    ".ogg":  ingest_audio,
    ".flac": ingest_audio,
}


def ingest_file(file_path: str):
    ext     = os.path.splitext(file_path)[1].lower()
    handler = SUPPORTED.get(ext)
    if not handler:
        raise ValueError(f"Unsupported file type: '{ext}'")
    print(f"\n[Ingestor] Processing: {os.path.basename(file_path)}")
    records = handler(file_path)
    print(f"[Ingestor] Done → {len(records)} chunks\n")
    return records