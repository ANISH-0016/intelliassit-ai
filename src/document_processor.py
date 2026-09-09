"""
document_processor.py
----------------------
Handles document ingestion for IntelliAssist AI.
Supports PDF, TXT, and DOCX files.
Extracts raw text and splits it into overlapping chunks suitable
for embedding + retrieval (RAG).
"""

from __future__ import annotations
import io
import re
import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A single retrievable unit of text."""
    chunk_id: str
    text: str
    source: str          # filename
    page: int | None = None
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def extract_text_from_pdf(file_bytes: bytes) -> List[tuple[int, str]]:
    """Returns list of (page_number, text) tuples."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def extract_text_from_docx(file_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # also grab table content
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)
    return "\n".join(parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Splits text into overlapping chunks by character count, trying to break
    on sentence boundaries where possible. Overlap preserves context across
    chunk edges, which improves retrieval recall.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[Chunk] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(_make_chunk(current, source, page))
            # start new chunk, carrying overlap from the tail of the previous
            overlap_text = current[-overlap:] if overlap and current else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(_make_chunk(current, source, page))

    # Fallback: if a single sentence is longer than chunk_size (e.g. no punctuation),
    # hard-split it.
    final_chunks: List[Chunk] = []
    for c in chunks:
        if len(c.text) <= chunk_size * 1.5:
            final_chunks.append(c)
        else:
            for i in range(0, len(c.text), chunk_size - overlap):
                piece = c.text[i:i + chunk_size]
                if piece.strip():
                    final_chunks.append(_make_chunk(piece, source, page))
    return final_chunks


def _make_chunk(text: str, source: str, page: int | None) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4())[:8],
        text=text.strip(),
        source=source,
        page=page,
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def process_document(filename: str, file_bytes: bytes) -> List[Chunk]:
    """
    Detects file type from extension, extracts text, and returns chunks
    ready to be embedded and indexed.
    """
    ext = filename.lower().split(".")[-1]
    all_chunks: List[Chunk] = []

    if ext == "pdf":
        pages = extract_text_from_pdf(file_bytes)
        for page_num, page_text in pages:
            all_chunks.extend(chunk_text(page_text, source=filename, page=page_num))
    elif ext == "docx":
        text = extract_text_from_docx(file_bytes)
        all_chunks.extend(chunk_text(text, source=filename))
    elif ext == "txt":
        text = extract_text_from_txt(file_bytes)
        all_chunks.extend(chunk_text(text, source=filename))
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use PDF, DOCX, or TXT.")

    return all_chunks
