"""
retrieval/query_engine.py
Uses Mistral via Ollama to answer questions from retrieved chunks.
"""

import ollama # type: ignore

DEFAULT_MODEL = "mistral"
DEFAULT_TOP_K = 5


def build_prompt(question: str, chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        loc = (f"Timestamp {chunk.get('timestamp','?')}"
               if chunk["file_type"] == "audio"
               else f"Page {chunk.get('page','?')}")
        context += (
            f"\n--- Source {i+1}: {chunk['source']} ({loc}) ---\n"
            f"{chunk['text'][:600]}\n"
        )

    return f"""You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I could not find relevant information in the uploaded files."
Always mention which document and page/timestamp your answer comes from.

====================
CONTEXT:
{context}
====================

QUESTION: {question}

ANSWER:"""


class QueryEngine:

    def __init__(self, vector_store, model: str = DEFAULT_MODEL):
        self.vector_store = vector_store
        self.model        = model
        self._check_ollama()

    def _check_ollama(self):
        try:
            models    = ollama.list()
            available = [m["name"] for m in models.get("models", [])]
            found     = any(self.model in m for m in available)
            if found:
                print(f"[QueryEngine] '{self.model}' is ready ✓")
            else:
                print(f"[QueryEngine] '{self.model}' not found. Run: ollama pull {self.model}")
        except Exception:
            print("[QueryEngine] Ollama not running! Run: ollama serve")

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        chunks = self.vector_store.query(question, top_k=top_k)

        if not chunks:
            return {"answer": "No documents indexed yet. Please upload files first.",
                    "citations": [], "chunks": []}

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": build_prompt(question, chunks)}],
                options={"temperature": 0.1, "num_predict": 512}
            )
            answer = response["message"]["content"]
        except Exception as e:
            answer = (f"Ollama error. Make sure it is running.\n"
                      f"Run: ollama serve\n\nError: {e}")

        seen, citations = set(), []
        for c in [ch["citation"] for ch in chunks]:
            if c not in seen:
                citations.append(c)
                seen.add(c)

        return {"answer": answer, "citations": citations, "chunks": chunks}

    def change_model(self, model_name: str):
        self.model = model_name
        self._check_ollama()