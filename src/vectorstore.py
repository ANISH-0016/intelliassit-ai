"""
vectorstore.py
--------------
Embedding + semantic search layer for IntelliAssist AI.

Two embedding backends are supported so the app NEVER breaks in a demo,
even with no internet access or GPU:

  1. "tfidf"  -> scikit-learn TF-IDF vectors (always available, zero downloads,
                 satisfies the "TF-IDF" item in the tech stack).
  2. "bert"   -> sentence-transformers (all-MiniLM-L6-v2) dense embeddings,
                 satisfies the "BERT Embeddings" item in the tech stack.
                 Used automatically if the package + model are available,
                 otherwise the app silently falls back to TF-IDF.

Similarity search is performed with FAISS when available (in-memory index),
falling back to plain NumPy cosine similarity if faiss isn't installed.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.document_processor import Chunk

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    Holds chunks + their embeddings and supports similarity search.
    Backend is chosen automatically: BERT embeddings if sentence-transformers
    is installed and the model can be loaded, else TF-IDF.
    """

    def __init__(self, prefer_bert: bool = True):
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.backend: str = "tfidf"
        self._tfidf: TfidfVectorizer | None = None
        self._st_model = None
        self._faiss_index = None

        if prefer_bert and _HAS_ST:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "bert"
            except Exception:
                self._st_model = None
                self.backend = "tfidf"

    # ------------------------------------------------------------------ #
    # Building the index
    # ------------------------------------------------------------------ #

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if not texts:
            self.embeddings = None
            return

        if self.backend == "bert" and self._st_model is not None:
            self.embeddings = np.asarray(
                self._st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            ).astype("float32")
        else:
            self.backend = "tfidf"
            self._tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
            matrix = self._tfidf.fit_transform(texts)
            self.embeddings = matrix.toarray().astype("float32")
            # normalize rows for cosine similarity via inner product
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms

        if _HAS_FAISS and self.embeddings is not None and len(self.embeddings) > 0:
            dim = self.embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.embeddings)
        else:
            self._faiss_index = None

    def add(self, chunks: List[Chunk]) -> None:
        """Incrementally add more chunks (rebuilds index for simplicity/correctness)."""
        self.build(self.chunks + chunks)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "bert" and self._st_model is not None:
            vec = self._st_model.encode([query], normalize_embeddings=True)
            return np.asarray(vec).astype("float32")
        else:
            vec = self._tfidf.transform([query]).toarray().astype("float32")
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            return vec / norm

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        q_vec = self._embed_query(query)
        top_k = min(top_k, len(self.chunks))

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q_vec, top_k)
            results = [
                (self.chunks[i], float(scores[0][rank]))
                for rank, i in enumerate(idxs[0]) if i != -1
            ]
        else:
            sims = cosine_similarity(q_vec, self.embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:top_k]
            results = [(self.chunks[i], float(sims[i])) for i in top_idx]

        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "store.pkl", "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "backend": self.backend,
                    "tfidf": self._tfidf,
                },
                f,
            )

    def load(self, path: str | Path) -> bool:
        path = Path(path) / "store.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        self.backend = data["backend"]
        self._tfidf = data["tfidf"]
        if _HAS_FAISS and self.embeddings is not None and len(self.embeddings) > 0:
            dim = self.embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.embeddings)
        return True

    def is_empty(self) -> bool:
        return not self.chunks
