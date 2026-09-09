"""
sentiment_intent.py
--------------------
Sentiment and intent analysis for IntelliAssist AI.

Sentiment: Hugging Face DistilBERT sentiment pipeline if available,
else TextBlob polarity (pure-Python, no downloads).

Intent: lightweight rule-based classifier over the user's chat query.
Categories: question, summarization_request, comparison, definition,
complaint/feedback, greeting, other.
"""

from __future__ import annotations
import re

_hf_sentiment = None
_hf_attempted = False


def _get_hf_sentiment():
    global _hf_sentiment, _hf_attempted
    if _hf_sentiment is not None or _hf_attempted:
        return _hf_sentiment
    _hf_attempted = True
    try:
        from transformers import pipeline
        _hf_sentiment = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    except Exception:
        _hf_sentiment = None
    return _hf_sentiment


def analyze_sentiment(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"label": "NEUTRAL", "score": 0.0, "method": "none"}

    pipe = _get_hf_sentiment()
    if pipe is not None:
        try:
            out = pipe(text[:512])[0]
            return {"label": out["label"], "score": float(out["score"]), "method": "distilbert"}
        except Exception:
            pass

    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        label = "POSITIVE" if polarity > 0.1 else "NEGATIVE" if polarity < -0.1 else "NEUTRAL"
        return {"label": label, "score": abs(polarity), "method": "textblob"}
    except Exception:
        return {"label": "NEUTRAL", "score": 0.0, "method": "unavailable"}


_INTENT_PATTERNS = [
    ("greeting", r"^(hi|hello|hey|good morning|good evening)\b"),
    ("summarization_request", r"\b(summar\w*|tl;?dr|overview|gist|brief me)\b"),
    ("comparison", r"\b(compare|versus|vs\.?|difference between|which is better)\b"),
    ("definition", r"\b(what is|what are|define|meaning of|explain)\b"),
    ("howto", r"\b(how (do|can|to)|steps to|guide for)\b"),
    ("complaint_feedback", r"\b(doesn'?t work|not working|error|issue|problem|wrong|bad|disappointed)\b"),
    ("thanks", r"\b(thanks|thank you|appreciate it)\b"),
]


def detect_intent(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "other"
    for label, pattern in _INTENT_PATTERNS:
        if re.search(pattern, lowered):
            return label
    if lowered.endswith("?") or lowered.startswith(("who", "when", "where", "why")):
        return "question"
    return "other"


def analyze(text: str) -> dict:
    """Convenience wrapper returning both sentiment and intent."""
    return {
        "sentiment": analyze_sentiment(text),
        "intent": detect_intent(text),
    }
