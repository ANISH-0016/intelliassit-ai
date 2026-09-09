"""
rag_engine.py
-------------
Retrieval-Augmented Generation pipeline for IntelliAssist AI.

Answer generation tries, in order:
  1. OpenAI  (if OPENAI_API_KEY is supplied)
  2. Google Gemini (if GEMINI_API_KEY is supplied)
  3. Local Hugging Face seq2seq model (google/flan-t5-base) if transformers
     is installed and the model can be loaded
  4. Extractive fallback (no external calls at all) -- guarantees the app
     always produces an answer, which is critical for a live demo.

Every answer returns the source chunks used, so the UI can display citations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from src.vectorstore import VectorStore
from src.document_processor import Chunk


@dataclass
class RAGAnswer:
    answer: str
    sources: List[Chunk]
    scores: List[float]
    backend_used: str


SYSTEM_PROMPT = (
    "You are IntelliAssist AI, a helpful assistant that answers questions "
    "ONLY using the provided document context. If the answer is not in the "
    "context, say you could not find it in the uploaded documents. Be "
    "concise and cite which document/page the information comes from when "
    "relevant."
)


def _build_context(results: List[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in results:
        loc = f"{chunk.source}" + (f" (page {chunk.page})" if chunk.page else "")
        parts.append(f"[Source: {loc}]\n{chunk.text}")
    return "\n\n".join(parts)


class RAGEngine:
    def __init__(self, vector_store: VectorStore, openai_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        self.vs = vector_store
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self._local_pipeline = None
        self._local_load_attempted = False

    # ------------------------------------------------------------------ #
    def answer(self, question: str, top_k: int = 4, chat_history: Optional[list] = None) -> RAGAnswer:
        results = self.vs.search(question, top_k=top_k)
        if not results:
            return RAGAnswer(
                answer="Please upload at least one document first so I have something to search.",
                sources=[], scores=[], backend_used="none",
            )

        context = _build_context(results)
        sources = [r[0] for r in results]
        scores = [r[1] for r in results]

        # Try backends in order of quality
        if self.openai_api_key:
            try:
                text = self._answer_openai(question, context, chat_history)
                return RAGAnswer(text, sources, scores, "openai")
            except Exception as e:
                last_err = f"OpenAI backend failed: {e}"
        else:
            last_err = None

        if self.gemini_api_key:
            try:
                text = self._answer_gemini(question, context, chat_history)
                return RAGAnswer(text, sources, scores, "gemini")
            except Exception as e:
                last_err = f"Gemini backend failed: {e}"

        try:
            text = self._answer_local_llm(question, context)
            if text:
                return RAGAnswer(text, sources, scores, "local-flan-t5")
        except Exception:
            pass

        # Guaranteed fallback -- always works, no downloads/network needed
        text = self._answer_extractive(question, results)
        return RAGAnswer(text, sources, scores, "extractive-fallback")

    # ------------------------------------------------------------------ #
    # Backend 1: OpenAI
    # ------------------------------------------------------------------ #
    def _answer_openai(self, question: str, context: str, chat_history) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for turn in chat_history[-4:]:
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        })
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # Backend 2: Gemini
    # ------------------------------------------------------------------ #
    def _answer_gemini(self, question: str, context: str, chat_history) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        resp = model.generate_content(prompt)
        return resp.text.strip()

    # ------------------------------------------------------------------ #
    # Backend 3: Local Hugging Face model
    # ------------------------------------------------------------------ #
    def _answer_local_llm(self, question: str, context: str) -> Optional[str]:
        if self._local_pipeline is None:
            if self._local_load_attempted:
                return None
            self._local_load_attempted = True
            from transformers import pipeline
            self._local_pipeline = pipeline(
                "text2text-generation", model="google/flan-t5-base", max_new_tokens=256
            )

        prompt = (
            f"Answer the question using only the context below. "
            f"If not found, say 'Not found in the documents.'\n\n"
            f"Context: {context[:3000]}\n\nQuestion: {question}\nAnswer:"
        )
        out = self._local_pipeline(prompt)
        return out[0]["generated_text"].strip()

    # ------------------------------------------------------------------ #
    # Backend 4: Extractive fallback (always available, no ML downloads)
    # ------------------------------------------------------------------ #
    def _answer_extractive(self, question: str, results: List[tuple[Chunk, float]]) -> str:
        best_chunk, best_score = results[0]
        snippet = best_chunk.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."

        loc = best_chunk.source + (f", page {best_chunk.page}" if best_chunk.page else "")
        header = (
            f"Based on the most relevant passage found ({loc}, "
            f"relevance {best_score:.0%}):\n\n"
        )
        return header + snippet
