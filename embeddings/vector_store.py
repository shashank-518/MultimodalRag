"""
embeddings/vector_store.py
Uses Mistral via Ollama for embeddings.
Supports PDF, Image, and Video chunk types.
"""

import os
import ollama  # type: ignore
import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore
from tqdm import tqdm

CHROMA_PATH     = "./data/VectorStore"
COLLECTION_NAME = "multimodal_rag"
EMBED_MODEL     = "mistral"
BATCH_SIZE      = 8

# Video sub-types handled by this store
VIDEO_TYPES = {"video_frame", "video_audio"}


def _embed(text: str) -> list:
    """Get embedding vector from Mistral via Ollama."""
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return resp["embedding"]
    except Exception as e:
        raise RuntimeError(
            f"Embedding failed. Make sure Ollama is running.\n"
            f"Run: ollama serve\n\nError: {e}"
        )


def _model_names(models_response) -> list[str]:
    """
    Normalise the ollama.list() response across old and new SDK versions.
    Old SDK  → list of dicts  with key 'name'
    New SDK  → list of Model objects with attribute .model  (or .name)
    """
    names = []
    for m in models_response.get("models", []):
        if isinstance(m, dict):
            names.append(m.get("name", "") or m.get("model", ""))
        else:
            # pydantic Model object
            names.append(getattr(m, "model", None) or getattr(m, "name", ""))
    return names


class VectorStore:

    def __init__(self):
        print(f"[VectorStore] Using Mistral embeddings via Ollama…")
        self._verify_ollama()

        os.makedirs(CHROMA_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Ready! {self.collection.count()} chunks indexed.")

    def _verify_ollama(self):
        try:
            available = _model_names(ollama.list())
            if any("mistral" in m for m in available):
                print("[VectorStore] Mistral ready ✓")
            else:
                print("[VectorStore] ⚠️  Mistral not found! Run: ollama pull mistral")
        except Exception:
            print("[VectorStore] ⚠️  Ollama not running! Run: ollama serve")

    # ── ADD ──────────────────────────────────────────

    def add_records(self, records: list):
        if not records:
            return 0

        existing_ids = set(self.collection.get(include=[])["ids"])
        new_records  = [r for r in records if str(r["id"]) not in existing_ids]

        if not new_records:
            print("[VectorStore] Already indexed — skipping.")
            return 0

        print(f"[VectorStore] Indexing {len(new_records)} chunks…")

        for i in tqdm(range(0, len(new_records), BATCH_SIZE), desc="Indexing"):
            batch      = new_records[i : i + BATCH_SIZE]
            texts      = [r["text"] for r in batch]
            ids        = [str(r["id"]) for r in batch]
            embeddings = [_embed(t) for t in texts]
            metadatas  = [{
                "source":      str(r.get("source",      "")),
                "file_type":   str(r.get("file_type",   "")),
                "page":        int(r.get("page",          1)),
                "total_pages": int(r.get("total_pages",   0)),
                "chunk":       int(r.get("chunk",          0)),
                "timestamp":   str(r.get("timestamp",    "")),
            } for r in batch]

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        print(f"[VectorStore] ✓ Done! Total: {self.collection.count()} chunks")
        return len(new_records)

    # ── QUERY ────────────────────────────────────────

    def query(self, question: str, top_k: int = 5):
        count = self.collection.count()
        if count == 0:
            return []

        q_emb   = _embed(question)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["documents"][0])):
            meta  = results["metadatas"][0][i]
            text  = results["documents"][0][i]
            score = round(1 - results["distances"][0][i], 4)
            ftype = meta.get("file_type", "")

            if ftype == "video_frame":
                citation = f"🎬 {meta['source']} — Frame @ {meta.get('timestamp','?')}"
            elif ftype == "video_audio":
                citation = f"🔊 {meta['source']} — Audio @ {meta.get('timestamp','?')}"
            elif ftype == "image":
                citation = f"🖼️ {meta['source']} — OCR"
            else:
                citation = f"📄 {meta['source']} — Page {meta.get('page','?')}"

            output.append({
                "text":      text,
                "source":    meta.get("source",    ""),
                "file_type": ftype,
                "page":      meta.get("page"),
                "timestamp": meta.get("timestamp"),
                "score":     score,
                "citation":  citation,
            })

        output.sort(key=lambda x: x["score"], reverse=True)
        return output

    # ── UTILS ────────────────────────────────────────

    def list_sources(self):
        try:
            if self.collection.count() == 0:
                return []
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            return sorted({m["source"] for m in metas if m.get("source")})
        except Exception as e:
            print(f"[VectorStore] list_sources error: {e}")
            return []

    def delete_source(self, filename: str):
        try:
            all_data = self.collection.get(include=["metadatas"])
            ids_del  = [
                all_data["ids"][i]
                for i, m in enumerate(all_data["metadatas"])
                if m.get("source") == filename
            ]
            if ids_del:
                self.collection.delete(ids=ids_del)
                print(f"[VectorStore] Deleted '{filename}'")
        except Exception as e:
            print(f"[VectorStore] delete error: {e}")

    def total_chunks(self):
        return self.collection.count()