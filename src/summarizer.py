"""
summarizer.py
-------------
Text summarisation for IntelliAssist AI.

Uses a Hugging Face abstractive summarisation pipeline (distilbart-cnn) when
available; otherwise falls back to a lightweight extractive summariser based
on TF-IDF sentence scoring (pure scikit-learn/numpy, no downloads required).
"""

from __future__ import annotations
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_hf_summarizer = None
_hf_attempted = False


def _get_hf_summarizer():
    global _hf_summarizer, _hf_attempted
    if _hf_summarizer is not None or _hf_attempted:
        return _hf_summarizer
    _hf_attempted = True
    try:
        from transformers import pipeline
        _hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        _hf_summarizer = None
    return _hf_summarizer


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """TF-IDF sentence-ranking summariser -- always works offline."""
    sentences = _split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = np.asarray(matrix.sum(axis=1)).flatten()

    # Reward sentences near the start (often contain thesis/intro info)
    position_bonus = np.linspace(1.15, 0.9, len(sentences))
    scores = scores * position_bonus

    top_idx = sorted(np.argsort(scores)[::-1][:num_sentences])
    return " ".join(sentences[i] for i in top_idx)


def summarize(text: str, num_sentences: int = 5, prefer_abstractive: bool = True) -> dict:
    """
    Returns {"summary": str, "method": "abstractive" | "extractive"}
    """
    text = text.strip()
    if not text:
        return {"summary": "", "method": "none"}

    if prefer_abstractive:
        summarizer = _get_hf_summarizer()
        if summarizer is not None:
            try:
                # chunk long text to respect model max input length
                chunks = [text[i:i + 3000] for i in range(0, len(text), 3000)][:4]
                pieces = []
                for c in chunks:
                    out = summarizer(c, max_length=140, min_length=30, do_sample=False)
                    pieces.append(out[0]["summary_text"])
                return {"summary": " ".join(pieces), "method": "abstractive (BART)"}
            except Exception:
                pass

    return {"summary": extractive_summary(text, num_sentences), "method": "extractive (TF-IDF)"}
