"""
ingestion/ingestor.py
Handles PDF, Images, and Video (frame OCR + audio transcription).
Lazy imports — a missing library only breaks that specific file type.
"""

import os
import shutil

# ── Windows: tell pytesseract where tesseract.exe is ──
import pytesseract  # type: ignore
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── CHUNKING ──────────────────────────────────────────

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
        import fitz  # type: ignore  (PyMuPDF)
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
            records.append(_rec(
                chunk, filename, "pdf",
                page_num + 1, len(doc), i,
                f"{filename}_p{page_num+1}_c{i}"
            ))
    doc.close()
    print(f"[PDF] {filename} → {len(records)} chunks")
    return records


# ── IMAGE ─────────────────────────────────────────────

def ingest_image(file_path: str):
    try:
        import pytesseract  # type: ignore
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Run: pip install pytesseract Pillow\n"
            "Also install Tesseract binary:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  Linux:   sudo apt install tesseract-ocr\n"
            "  Mac:     brew install tesseract"
        )

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
        records.append(_rec(
            chunk, filename, "image",
            1, 1, i,
            f"{filename}_img_c{i}"
        ))

    print(f"[IMAGE] {filename} → {len(records)} chunks")
    return records


# ── VIDEO ─────────────────────────────────────────────

_whisper_model = None


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _extract_audio_from_video(video_path: str, audio_out: str):
    """Use ffmpeg to rip the audio track to a wav file."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg not found.\n"
            "  Windows: https://ffmpeg.org/download.html (add to PATH)\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "  Mac:     brew install ffmpeg"
        )
    ret = os.system(
        f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le '
        f'-ar 16000 -ac 1 "{audio_out}" -loglevel error'
    )
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed extracting audio from {video_path}")


def _transcribe_audio(audio_path: str) -> list:
    """Transcribe audio with Whisper; return list of segment dicts."""
    try:
        import whisper  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    global _whisper_model
    if _whisper_model is None:
        print("[Whisper] Loading model (first time may be slow)…")
        _whisper_model = whisper.load_model("base")
        print("[Whisper] Model ready ✓")

    result = _whisper_model.transcribe(audio_path, verbose=False, fp16=False)
    return result.get("segments", [])


def _ocr_video_frames(video_path: str, every_n_seconds: int = 30) -> list:
    """
    Sample one frame every `every_n_seconds` seconds, run OCR on it,
    return list of (timestamp_str, ocr_text) tuples.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        raise ImportError("Run: pip install opencv-python")

    try:
        import pytesseract  # type: ignore
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install pytesseract Pillow")

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    step   = int(fps * every_n_seconds)
    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts  = _fmt_ts(frame_idx / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            txt = pytesseract.image_to_string(pil).strip()
            if txt:
                frame_results.append((ts, txt))
        frame_idx += 1

    cap.release()
    return frame_results


def ingest_video(file_path: str):
    filename = os.path.basename(file_path)
    records  = []
    idx      = 0

    # ── 1. OCR on sampled frames ──────────────────────
    print(f"[Video] Extracting text from frames of {filename}…")
    try:
        frame_data = _ocr_video_frames(file_path)
        for ts, text in frame_data:
            for i, chunk in enumerate(split_into_chunks(text)):
                records.append(_rec(
                    chunk, filename, "video_frame",
                    idx + 1, 0, i,
                    f"{filename}_frame_{ts}_c{i}",
                    timestamp=ts
                ))
                idx += 1
        print(f"[Video] {len(frame_data)} frames with text extracted.")
    except (ImportError, Exception) as e:
        print(f"[Video] Frame OCR skipped: {e}")

    # ── 2. Audio transcription ────────────────────────
    print(f"[Video] Transcribing audio of {filename}…")
    tmp_audio = file_path + "_tmp_audio.wav"
    try:
        _extract_audio_from_video(file_path, tmp_audio)
        segments = _transcribe_audio(tmp_audio)

        buffer       = ""
        buffer_start = 0.0
        first        = True
        seg_idx      = 0

        for seg in segments:
            t = seg.get("text", "").strip()
            if not t:
                continue
            if first:
                buffer_start = seg.get("start", 0.0)
                first = False
            buffer += " " + t

            if len(buffer.split()) >= 300:
                records.append(_rec(
                    buffer.strip(), filename, "video_audio",
                    seg_idx + 1, 0, seg_idx,
                    f"{filename}_audio_t{seg_idx}",
                    timestamp=_fmt_ts(buffer_start)
                ))
                seg_idx += 1
                buffer  = ""
                first   = True

        if buffer.strip():
            records.append(_rec(
                buffer.strip(), filename, "video_audio",
                seg_idx + 1, 0, seg_idx,
                f"{filename}_audio_t{seg_idx}",
                timestamp=_fmt_ts(buffer_start)
            ))

        # patch total_pages
        audio_recs = [r for r in records if r["file_type"] == "video_audio"]
        for r in audio_recs:
            r["total_pages"] = len(audio_recs)

        print(f"[Video] Audio transcription → {len(audio_recs)} chunks")

    except EnvironmentError as e:
        print(f"[Video] Audio transcription skipped (ffmpeg missing): {e}")
    except Exception as e:
        print(f"[Video] Audio transcription error: {e}")
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    print(f"[Video] {filename} → {len(records)} total chunks")
    return records


# ── ROUTER ────────────────────────────────────────────

SUPPORTED = {
    # PDF
    ".pdf":  ingest_pdf,
    # Images
    ".png":  ingest_image,
    ".jpg":  ingest_image,
    ".jpeg": ingest_image,
    ".tiff": ingest_image,
    ".bmp":  ingest_image,
    ".webp": ingest_image,
    # Video
    ".mp4":  ingest_video,
    ".mov":  ingest_video,
    ".avi":  ingest_video,
    ".mkv":  ingest_video,
    ".webm": ingest_video,
    ".flv":  ingest_video,
}


def ingest_file(file_path: str):
    ext     = os.path.splitext(file_path)[1].lower()
    handler = SUPPORTED.get(ext)
    if not handler:
        raise ValueError(
            f"Unsupported file type: '{ext}'\n"
            f"Supported: PDF, Images (png/jpg/jpeg/tiff/bmp/webp), "
            f"Video (mp4/mov/avi/mkv/webm/flv)"
        )
    print(f"\n[Ingestor] Processing: {os.path.basename(file_path)}")
    records = handler(file_path)
    print(f"[Ingestor] Done → {len(records)} chunks\n")
    return records