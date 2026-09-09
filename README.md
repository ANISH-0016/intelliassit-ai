# 🧠 IntelliAssist AI — Smart Document AI Assistant

An AI-powered chatbot that lets you upload documents (PDF / DOCX / TXT) and ask
questions about them using **Retrieval-Augmented Generation (RAG)**. Built as an
Artificial Intelligence Major Capstone Project.



---

## ✨ Features

| Feature | How it's implemented |
|---|---|
| Document Upload (PDF/TXT/DOCX) | `pypdf`, `python-docx` |
| Semantic Search | TF-IDF (scikit-learn) → auto-upgrades to BERT (`sentence-transformers`) |
| Vector Database | FAISS (auto-upgrades from NumPy cosine similarity) |
| RAG Chatbot | Custom retrieval + generation pipeline (OpenAI / Gemini / local Flan-T5 / extractive) |
| Summarisation | Abstractive (BART) with TF-IDF extractive fallback |
| Sentiment & Intent Analysis | DistilBERT / TextBlob + rule-based intent classifier |
| Source Citation | Every answer links back to filename + page number |
| Conversation History | Session-based chat log + analytics dashboard |
| Web Interface | Streamlit |

---




## 🧩 Architecture / RAG Workflow

```
 Upload File → Extract Text → Chunk (800 chars, 150 overlap)
        │
        ▼
 Generate Embeddings (TF-IDF or BERT) → Store in Vector Index (FAISS)
        │
        ▼
 User Question → Embed Query → Similarity Search (top-k chunks)
        │
        ▼
 Build Context from retrieved chunks
        │
        ▼
 Generate Answer:
   1) OpenAI (if key set)
   2) Gemini (if key set)
   3) Local Flan-T5 (if transformers installed)
   4) Extractive fallback (always works)
        │
        ▼
 Display Answer + Source Citations + Sentiment/Intent of the query
```

---




---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend / AI:** Python, custom RAG pipeline, Hugging Face, OpenAI/Gemini API (optional)
- **NLP:** TF-IDF, BERT Embeddings (sentence-transformers)
- **Vector DB:** FAISS (NumPy cosine-similarity fallback)
- **Deployment:** Streamlit Cloud / Render

---

